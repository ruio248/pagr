"""Compare online/target Critic action gradients with split MC references.

This read-only PointObstacle audit reuses saved Learned-QAM snapshots. It does
not train or update Actor/Critic parameters. At each frozen snapshot it uses the
same states, Actor noise, Flow noise, and action endpoints for both critics and
two independent current-policy Monte Carlo score-gradient estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNNER = ROOT / "scripts" / "run_point_gradient_magnitude_direction.py"
SPEC = importlib.util.spec_from_file_location("point_gradient_magnitude_direction", SOURCE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load PointObstacle experiment runner: {SOURCE_RUNNER}")
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)
M, CORE, longrun = experiment.M, experiment.CORE, experiment.longrun
torch.set_num_threads(4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def rms(x: torch.Tensor) -> float:
    return float(x.square().mean().sqrt())


def fit_nonnegative_scale(source: torch.Tensor, target: torch.Tensor) -> float:
    denominator = float(source.square().sum())
    if denominator <= 1e-24:
        return 0.0
    return max(0.0, float((source * target).sum()) / denominator)


def validate_mc_semantics(source_path: Path, point_path: Path) -> dict:
    """Record audited estimator semantics and exact source hashes."""
    source = source_path.read_text()
    point = point_path.read_text()
    independent_start = source.index("def independent_mc(")
    independent_end = source.index("\n\nclass MonteCarloReference", independent_start)
    mc_source = source[independent_start:independent_end]
    checks = {
        "first_action_is_fixed_query_action": "action = np.broadcast_to(np.asarray(a" in mc_source,
        "future_actions_use_supplied_current_actor": "sample_actor_tensor(policy" in mc_source,
        "discounted_return_uses_gamma_from_query_time": "c.gamma**(j-k)*reward" in mc_source,
        "terminal_mask_is_task_transition_done": "task.transition(j, state[ids], action, noise)" in mc_source,
        "gaussian_score_and_tanh_chain_rule_present": (
            "first_noise-first_noise.mean(0)" in mc_source
            and "chain = c.delta*(1-np.tanh(a)**2)" in mc_source
            and "c.sigma_env" in mc_source
        ),
        "mc_estimator_does_not_read_critic": "critic" not in mc_source,
        "online_critic_gradient_uses_critic_mean": "q = critic(k, y, a).mean(-1)" in point,
        "target_critic_gradient_uses_same_action_derivative": (
            "q = critic(k, y, a).mean(-1)" in point
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"MC/critic semantic source check failed: {failed}")
    checks["interpretation"] = (
        "MC estimates the current-policy action-value gradient for a fixed query action; "
        "it is a finite-sample reference, not an exact oracle. The target Critic is an EMA "
        "estimate and is not automatically the same object as Q^pi_current."
    )
    checks["source_hashes"] = {
        "run_point_long_training.py": sha256(source_path),
        "run_point_gradient_magnitude_direction.py": sha256(point_path),
        "point_obstacle_notebook.py": sha256(ROOT / "notebooks" / "02_qam_point_obstacle.py"),
    }
    return checks


def run(output: Path, seed: int = 59001, snapshots=(0, 50, 100), episodes: int = 8192,
        device: str | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "configs" / "point_gradient_magnitude_direction.json"
    protocol = json.loads(protocol_path.read_text())
    if seed != 59001 or snapshots != (0, 50, 100) or episodes != 8192:
        raise ValueError("this first audit is frozen to seed 59001, N=0/50/100, and 8192 MC episodes")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    c = experiment.load_effective_config(protocol)
    source, seed_dir, input_dir, _source_config, input_manifest = experiment.validate_source(c, protocol)
    init_path = input_dir / protocol["initialization_source"].split("/", 1)[1]
    train_path = input_dir / protocol["train_data_source"].split("/", 1)[1]
    init = torch.load(init_path, map_location="cpu", weights_only=True)
    base = CORE.ConditionalVelocity(c)
    base.load_state_dict(init["base"])
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    train_np = np.load(train_path, allow_pickle=False)
    train = M.as_torch({key: train_np[key] for key in train_np.files})
    diag = experiment.diagnostic_batch(train, c, seed)
    task = M.PointObstacle(c)

    source_code = ROOT / "scripts" / "run_point_long_training.py"
    point_code = ROOT / "scripts" / "run_point_gradient_magnitude_direction.py"
    semantic = validate_mc_semantics(source_code, point_code)
    semantic.update({
        "seed": seed,
        "snapshots": list(snapshots),
        "reference_episodes_per_split_per_endpoint": episodes,
        "diagnostic_batch_size": len(diag["k"]),
        "endpoint_generation": "same fixed state, z, and Flow-noise batch at each snapshot",
        "critic_source": "saved learned-N checkpoint online critic and EMA target critic",
        "reference_source": "two independent current-policy IID MC score estimates, averaged",
        "no_training_or_optimizer_updates": True,
    })
    write_json(output / "oracle-semantic-validation.json", semantic)

    rows = []
    started = time.monotonic()
    for N in snapshots:
        checkpoint = seed_dir / f"learned-N{N}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actor = CORE.ConditionalVelocity(c)
        actor.load_state_dict(saved["actor"])
        actor.eval()
        online = CORE.Critic(c)
        target = CORE.Critic(c)
        online.load_state_dict(saved["critic"])
        target.load_state_dict(saved["target"])
        paths, _ = CORE.rollout(actor, base, c, diag["k"], diag["y"], diag["z"], diag["flow"])
        endpoint = paths[:, -1]
        g_online = experiment.critic_action_gradient(online, diag["k"], diag["y"], endpoint)
        g_target = experiment.critic_action_gradient(target, diag["k"], diag["y"], endpoint)
        g_a, se_a, _, _, seconds_a = experiment.mc_reference_batch(
            task, actor, base, c, seed, N, diag["k"], diag["y"], endpoint,
            episodes, device, role=820000)
        g_b, se_b, _, _, seconds_b = experiment.mc_reference_batch(
            task, actor, base, c, seed, N, diag["k"], diag["y"], endpoint,
            episodes, device, role=840000)
        g_ref = 0.5 * (g_a + g_b)
        online_scale = fit_nonnegative_scale(g_online, g_ref)
        target_scale = fit_nonnegative_scale(g_target, g_ref)
        g_online_scaled = online_scale * g_online
        g_target_scaled = target_scale * g_target
        se_ref = 0.5 * np.sqrt(np.square(se_a) + np.square(se_b))
        split_diff = g_a - g_b
        for i in range(len(diag["k"])):
            ref_i = g_ref[i:i+1]
            online_i = g_online[i:i+1]
            target_i = g_target[i:i+1]
            rows.append({
                "N": int(N), "sample_index": int(i), "k": int(diag["k"][i]),
                "actor_hash": M.model_hash(actor),
                "online_gradient_norm": float(online_i.norm()),
                "target_gradient_norm": float(target_i.norm()),
                "mc_reference_gradient_norm": float(ref_i.norm()),
                "online_vs_mc_cosine": float(experiment.cosine_rows(online_i, ref_i)[0]),
                "target_vs_mc_cosine": float(experiment.cosine_rows(target_i, ref_i)[0]),
                "online_mc_error_norm": float((online_i-ref_i).norm()),
                "target_mc_error_norm": float((target_i-ref_i).norm()),
                "online_mc_error_rms_coordinate": rms(online_i-ref_i),
                "target_mc_error_rms_coordinate": rms(target_i-ref_i),
                "online_best_nonnegative_batch_scale": online_scale,
                "target_best_nonnegative_batch_scale": target_scale,
                "online_scaled_mc_error_norm": float((online_scale*online_i-ref_i).norm()),
                "target_scaled_mc_error_norm": float((target_scale*target_i-ref_i).norm()),
                "mc_split_difference_norm": float(split_diff[i].norm()),
                "mc_reference_se_l2": float(np.linalg.norm(se_ref[i])),
                "mc_reference_se_x": float(se_ref[i, 0]),
                "mc_reference_se_y": float(se_ref[i, 1]),
                "target_closer_than_online": bool((target_i-ref_i).norm() < (online_i-ref_i).norm()),
            })
        print(f"AUDIT N={N} complete; MC seconds={seconds_a+seconds_b:.2f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "online-target-mc-gradient.csv", index=False)
    summary_rows = []
    for N, group in frame.groupby("N", sort=True):
        online_error = group["online_mc_error_norm"].to_numpy()
        target_error = group["target_mc_error_norm"].to_numpy()
        online_scaled_error = group["online_scaled_mc_error_norm"].to_numpy()
        target_scaled_error = group["target_scaled_mc_error_norm"].to_numpy()
        ref_norm = group["mc_reference_gradient_norm"].to_numpy()
        ref_norm_rms = float(np.sqrt(np.mean(ref_norm**2)))
        online_error_rms = float(np.sqrt(np.mean(online_error**2)))
        target_error_rms = float(np.sqrt(np.mean(target_error**2)))
        online_scaled_error_rms = float(np.sqrt(np.mean(online_scaled_error**2)))
        target_scaled_error_rms = float(np.sqrt(np.mean(target_scaled_error**2)))
        summary_rows.append({
            "N": int(N), "samples": int(len(group)),
            "online_gradient_norm_mean": float(group.online_gradient_norm.mean()),
            "target_gradient_norm_mean": float(group.target_gradient_norm.mean()),
            "mc_gradient_norm_mean": float(group.mc_reference_gradient_norm.mean()),
            "mc_gradient_norm_rms": ref_norm_rms,
            "online_mc_error_rms": online_error_rms,
            "target_mc_error_rms": target_error_rms,
            "online_mc_relative_rms": online_error_rms / max(ref_norm_rms, 1e-12),
            "target_mc_relative_rms": target_error_rms / max(ref_norm_rms, 1e-12),
            "online_mc_cosine_mean": float(group.online_vs_mc_cosine.mean()),
            "target_mc_cosine_mean": float(group.target_vs_mc_cosine.mean()),
            "online_best_nonnegative_batch_scale": float(group.online_best_nonnegative_batch_scale.iloc[0]),
            "target_best_nonnegative_batch_scale": float(group.target_best_nonnegative_batch_scale.iloc[0]),
            "online_scaled_mc_error_rms": online_scaled_error_rms,
            "target_scaled_mc_error_rms": target_scaled_error_rms,
            "online_scaled_mc_relative_rms": online_scaled_error_rms / max(ref_norm_rms, 1e-12),
            "target_scaled_mc_relative_rms": target_scaled_error_rms / max(ref_norm_rms, 1e-12),
            "target_closer_fraction": float(group.target_closer_than_online.mean()),
            "mc_split_difference_rms": float(np.sqrt(np.mean(group.mc_split_difference_norm**2))),
            "mc_reference_se_l2_mean": float(group.mc_reference_se_l2.mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "online-target-mc-summary.csv", index=False)
    summary_columns = list(summary.columns)
    summary_markdown = [
        "| " + " | ".join(summary_columns) + " |",
        "|" + "|".join(["---"] * len(summary_columns)) + "|",
    ]
    for row in summary.itertuples(index=False, name=None):
        formatted = [str(value) if isinstance(value, (int, np.integer)) else f"{float(value):.5g}"
                     for value in row]
        summary_markdown.append("| " + " | ".join(formatted) + " |")
    lines = [
        "# PointObstacle online vs target Critic gradient audit",
        "",
        "Read-only audit of saved seed-59001 Learned-QAM snapshots. The same states, Actor noise, Flow noise, and endpoints are used for online Critic, EMA target Critic, and two independent 8192-rollout MC references.",
        "",
        "The MC reference fixes the query action for the first transition, follows the saved current Actor thereafter, uses the task transition/termination logic and discount, and applies the Gaussian score with the tanh action chain rule. It does not read either Critic. It is a finite-sample reference, not an exact oracle.",
        "",
        "## Summary",
        "",
        *summary_markdown,
        "",
        "## Interpretation guardrails",
        "",
        "- Smaller MC error and higher cosine are descriptive; the averaged MC estimate still has finite sampling error.",
        "- The target Critic is an EMA estimate, not automatically the exact current-policy Q gradient.",
        "- This audit does not update a model and does not select beta or a checkpoint.",
        "- If the target critic is materially closer to MC, the main comparison must first decide whether it is intended to represent canonical QAM for this toy; do not silently replace the existing baseline.",
        "",
        "## Artifacts",
        "",
        "- `online-target-mc-gradient.csv`: endpoint-level critic/reference diagnostics.",
        "- `online-target-mc-summary.csv`: snapshot summaries.",
        "- `oracle-semantic-validation.json`: source-checked semantics and hashes.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    write_json(output / "complete.json", {
        "status": "complete", "seed": seed, "snapshots": list(snapshots),
        "mc_episodes_per_split_per_endpoint": episodes,
        "reference_calls": len(snapshots) * 2 * len(diag["k"]),
        "duration_seconds": time.monotonic()-started,
        "device": device,
        "source_run": protocol["source_run"],
        "input_hashes": {
            "initialization": sha256(init_path), "train_transitions": sha256(train_path),
        },
    })
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=59001)
    parser.add_argument("--episodes", type=int, default=8192)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.output.resolve(), seed=args.seed, episodes=args.episodes, device=args.device)


if __name__ == "__main__":
    main()
