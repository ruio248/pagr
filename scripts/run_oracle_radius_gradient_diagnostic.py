"""Privileged radius and gradient-direction diagnostics for obstacle QAM.

The training diagnostic gives the pathwise robust solver only a per-sample
MC estimate of ||g_reference - g_learned||; it does not pass the error vector.
The direction/magnitude decomposition reuses held-out MC audit files and is
read-only. Neither component is a deployable algorithm or a general guarantee.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import secrets
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_point_long_training as longrun
import terminal_consistent_robust_qam as robust

M = longrun.M
CORE = M.core
PROTOCOL_PATH = ROOT / "configs" / "oracle_radius_gradient_diagnostic.json"
BASE_PROTOCOL_PATH = ROOT / "configs" / "point_long_training_mc.json"
torch.set_num_threads(min(4, torch.get_num_threads()))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def effective_config(protocol):
    c, _ = M.load_config()
    return replace(c, actor_updates=protocol["actor_updates"],
                   curve_updates=tuple(protocol["checkpoints"]),
                   curve_episodes=protocol["curve_episodes"],
                   final_episodes=protocol["final_episodes"],
                   reference_gradient_samples=protocol["reference_mc_episodes"])


def load_initialization(seed, c, output, protocol):
    source_root = ROOT / protocol["paired_baseline_run"] / f"seed-{seed}"
    if not (source_root / "complete.json").exists():
        raise FileNotFoundError(f"incomplete reference seed run: {source_root}")
    metadata_path = source_root.parent / "frozen-protocol.json"
    prior = json.loads(metadata_path.read_text())["config"]
    for key, value in asdict(c).items():
        if not key.startswith("reference_") and json.loads(json.dumps(value)) != prior[key]:
            raise RuntimeError(f"QAM initialization/config mismatch for {key}")
    required = ("shared-initialization.pt", "train-transitions.npz",
                "valid-transitions.npz")
    for name in required:
        if not (source_root / name).exists():
            raise FileNotFoundError(source_root / name)
    for name in required:
        target = output / name
        if not target.exists():
            shutil.copy2(source_root / name, target)
    archive = np.load(output / "train-transitions.npz")
    train = M.as_torch({key: archive[key] for key in archive.files})
    saved = torch.load(output / "shared-initialization.pt", map_location="cpu",
                       weights_only=True)
    base = CORE.ConditionalVelocity(c)
    base.load_state_dict(saved["base"])
    critic = CORE.Critic(c)
    critic.load_state_dict(saved["critic"])
    target = CORE.Critic(c)
    target.load_state_dict(saved["target"])
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return train, base, critic, target, source_root


def new_state(base, critic, target, c, seed):
    actor = copy.deepcopy(base)
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    return dict(actor=actor, critic=copy.deepcopy(critic),
                target=copy.deepcopy(target),
                aopt=torch.optim.Adam(actor.parameters(), lr=c.actor_lr),
                copt=torch.optim.Adam(critic.parameters(), lr=c.critic_lr),
                rng=CORE.generator(M.seed_for(seed, 701)))


def save_state(path, state, seed, completed):
    data = dict(seed=seed, completed=completed,
                actor=state["actor"].state_dict(),
                critic=state["critic"].state_dict(),
                target=state["target"].state_dict(),
                actor_optimizer=state["aopt"].state_dict(),
                critic_optimizer=state["copt"].state_dict(),
                rng=state["rng"].get_state())
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def restore_state(path, state):
    data = torch.load(path, map_location="cpu", weights_only=True)
    for name in ("actor", "critic", "target"):
        state[name].load_state_dict(data[name])
    state["aopt"].load_state_dict(data["actor_optimizer"])
    state["copt"].load_state_dict(data["critic_optimizer"])
    state["rng"].set_state(data["rng"])
    return int(data["completed"])


def mc_reference_batch(task, actor, base, c, seed, update, k, positions,
                       actions, episodes, device):
    """Independent current-policy score gradients for a batch of endpoints."""
    frozen_actor = copy.deepcopy(actor).to(device).eval()
    gradients, standard_errors = [], []
    for index in range(len(k)):
        mc_seed = M.seed_for(seed, 600000 + update * 100 + index)
        result = longrun.independent_mc(
            task, frozen_actor, base, int(k[index]), positions[index].numpy(),
            actions[index].detach().numpy(), episodes, mc_seed, device)
        gradients.append(result["gradient"])
        standard_errors.append(result["gradient_se"])
    return (torch.as_tensor(np.asarray(gradients), dtype=actions.dtype),
            np.asarray(standard_errors, dtype=np.float64))


def learned_gradient(critic, k, y, action):
    with torch.enable_grad():
        endpoint = action.detach().requires_grad_(True)
        q = critic(k, y, endpoint).mean(-1)
        gradient = torch.autograd.grad(q.sum(), endpoint)[0]
    return gradient.detach()


def oracle_radius_actor_step(state, base, task, c, seed, update,
                             k, y, z, flow, protocol, device):
    actor, critic = state["actor"], state["critic"]
    paths, _ = CORE.rollout(actor, base, c, k, y, z, flow)
    endpoint = paths[:, -1]
    g_learned = learned_gradient(critic, k, y, endpoint)
    g_reference, reference_se = mc_reference_batch(
        task, actor, base, c, seed, update, k, y, endpoint,
        protocol["reference_mc_episodes"], device)
    epsilon = (g_reference - g_learned).norm(dim=-1)

    # The robust update receives only the nonnegative error radii. The MC error
    # vector itself is not passed into the quadratic solver or costate target.
    terminal = -c.beta * g_learned
    p_hat = CORE.lean_adjoint(base, c, k, y, paths, terminal)
    robust_p, q_star, robust_info = robust.pathwise_terminal_robust_target(
        base, c, k, y, paths, p_hat, epsilon)
    q_grad, loss = CORE.actor_gradient(actor, base, c, k, y, paths, robust_p)
    CORE.set_gradient(actor, q_grad)
    clipped = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), c.actor_clip, error_if_nonfinite=True)
    state["aopt"].step()
    actor.zero_grad(set_to_none=True)
    diagnostics = dict(
        matching_loss=loss,
        matching_gradient_norm=float(q_grad.norm()),
        pre_clip_norm=float(clipped),
        oracle_radius_mean=float(epsilon.mean()),
        oracle_radius_median=float(epsilon.median()),
        oracle_radius_p90=float(torch.quantile(epsilon, .9)),
        learned_reference_error_mean=float((g_reference-g_learned).norm(dim=-1).mean()),
        reference_gradient_se_rms=float(np.sqrt(np.mean(reference_se ** 2))),
        **robust_info)
    return diagnostics


def load_resume(path, state):
    if path.exists():
        return restore_state(path, state)
    return 0


def train_oracle_radius_seed(seed, c, protocol, output):
    folder = output / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    train, base, initial_critic, initial_target, source = load_initialization(
        seed, c, folder, protocol)
    task = M.PointObstacle(c)
    state = new_state(base, initial_critic, initial_target, c, seed)
    latest = folder / "oracle-radius-latest.pt"
    completed = load_resume(latest, state)
    curve_path = folder / "oracle-radius-curves.csv"
    train_path = folder / "oracle-radius-train.csv"
    curves = pd.read_csv(curve_path).to_dict("records") if curve_path.exists() else []
    logs = pd.read_csv(train_path).to_dict("records") if train_path.exists() else []
    curve_bank = M.noise_bank(c, protocol["curve_episodes"], M.seed_for(seed, 501))
    final_bank = M.noise_bank(c, protocol["final_episodes"], M.seed_for(seed, 502))
    checkpoints = set(protocol["checkpoints"])

    for update in range(completed, protocol["actor_updates"] + 1):
        if update in checkpoints and not any(int(row["N"]) == update for row in curves):
            stats = M.evaluate(task, state["actor"], base, curve_bank)
            curves.append(dict(seed=seed, method="oracle_radius_robust", N=update, **stats))
            pd.DataFrame(curves).to_csv(curve_path, index=False)
            save_state(folder / f"oracle-radius-N{update}.pt", state, seed, update)
            print("RADIUS CURVE", seed, update, json.dumps(stats), flush=True)
        if update == protocol["actor_updates"]:
            break

        rng = state["rng"]
        critic_ids = torch.randint(len(train["k"]), (c.critic_batch,), generator=rng)
        td_loss, q_mean = M.td_step(state["critic"], state["target"],
                                    state["copt"], train, critic_ids, rng,
                                    c, state["actor"], base)
        actor_ids = torch.randint(len(train["k"]), (c.actor_batch,), generator=rng)
        k, y = train["k"][actor_ids], train["y"][actor_ids]
        z = torch.randn((len(actor_ids), c.dimension), generator=rng)
        flow = torch.randn((len(actor_ids), c.flow_steps, c.dimension), generator=rng)
        start = time.monotonic()
        info = oracle_radius_actor_step(state, base, task, c, seed, update,
                                        k, y, z, flow, protocol, "cpu")
        log = dict(seed=seed, method="oracle_radius_robust", N=update+1,
                   td_loss=td_loss, q_mean=q_mean,
                   update_seconds=time.monotonic()-start, **info)
        logs.append(log)
        if (update + 1) % 10 == 0:
            pd.DataFrame(logs).to_csv(train_path, index=False)
            save_state(latest, state, seed, update + 1)
            print("RADIUS UPDATE", seed, update + 1,
                  json.dumps({key: log[key] for key in (
                      "update_seconds", "oracle_radius_mean",
                      "reference_gradient_se_rms", "robust_to_nominal_drift_ratio")}),
                  flush=True)
        elif update + 1 == protocol["actor_updates"]:
            pd.DataFrame(logs).to_csv(train_path, index=False)
            save_state(latest, state, seed, update + 1)

    if not (folder / "oracle-radius-N300.pt").exists():
        save_state(folder / "oracle-radius-N300.pt", state, seed,
                   protocol["actor_updates"])
    stats, _, robust_status = M.evaluate(task, state["actor"], base, final_bank,
                                         trajectories=True)
    final = dict(seed=seed, method="oracle_radius_robust", N=protocol["actor_updates"], **stats)
    pd.DataFrame([final]).to_csv(folder / "oracle-radius-final.csv", index=False)

    baseline_source = pd.read_csv(source.parent / "formal-final.csv")
    baseline = baseline_source[(baseline_source.seed == seed) &
                               (baseline_source.method == "lean_am")].iloc[0].to_dict()
    baseline_checkpoint = torch.load(source / "lean_am" / "lean_am-N300.pt",
                                     map_location="cpu", weights_only=True)
    baseline_actor = CORE.ConditionalVelocity(c)
    baseline_actor.load_state_dict(baseline_checkpoint["actor"])
    for parameter in baseline_actor.parameters():
        parameter.requires_grad_(False)
    baseline_stats, _, baseline_status = M.evaluate(task, baseline_actor, base,
        final_bank, trajectories=True)
    for metric in ("success_rate", "collision_rate", "discounted_return"):
        if not np.isclose(baseline_stats[metric], float(baseline[metric]), atol=1e-12):
            raise RuntimeError(f"baseline paired evaluation mismatch for {metric}")
    if len(baseline_status) != len(robust_status):
        raise RuntimeError("paired final evaluation length mismatch")
    paired = pd.DataFrame(dict(seed=seed, episode=np.arange(len(robust_status)),
                               lean_am_status=baseline_status,
                               oracle_radius_status=robust_status))
    paired.to_csv(folder / "paired-final-outcomes.csv", index=False)
    paired_summary = dict(
        seed=seed,
        lean_failure_to_robust_success=int(((baseline_status != "success") &
                                            (robust_status == "success")).sum()),
        lean_success_to_robust_failure=int(((baseline_status == "success") &
                                            (robust_status != "success")).sum()),
        both_success=int(((baseline_status == "success") &
                          (robust_status == "success")).sum()),
        both_failure=int(((baseline_status != "success") &
                          (robust_status != "success")).sum()))
    write_json(folder / "complete.json", dict(seed=seed, final=final,
        baseline=baseline, paired=paired_summary,
        source_initialization=str(source), completed_at=time.time()))
    print("RADIUS SEED COMPLETE", seed, json.dumps(dict(final=final,
          paired=paired_summary)), flush=True)
    return dict(**baseline, source_method="lean_am"), final, paired_summary


def summarize_radius(rows, pairs, output, protocol):
    final = pd.DataFrame(rows)
    final.to_csv(output / "oracle-radius-final-all.csv", index=False)
    aggregate = final.groupby("method").agg(
        success_mean=("success_rate", "mean"), success_sd=("success_rate", "std"),
        collision_mean=("collision_rate", "mean"), collision_sd=("collision_rate", "std"),
        return_mean=("discounted_return", "mean"), return_sd=("discounted_return", "std"))
    aggregate.to_csv(output / "oracle-radius-aggregate.csv")
    wide = final.pivot(index="seed", columns="method", values="success_rate")
    differences = 100 * (wide["oracle_radius_robust"] - wide["learned"])
    result = dict(
        seed_differences_pp={str(int(key)): float(value) for key, value in differences.items()},
        mean_difference_pp=float(differences.mean()),
        seed_sd_pp=float(differences.std(ddof=1)),
        wins=int((differences > 0).sum()), ties=int((differences == 0).sum()),
        losses=int((differences < 0).sum()),
        paired_transitions={key: sum(row[key] for row in pairs)
                            for key in pairs[0] if key != "seed"})
    write_json(output / "oracle-radius-paired-summary.json", result)

    lines = [
        "# Privileged oracle-radius robust QAM: exploratory diagnostic", "",
        "For each sampled Actor path, the endpoint Critic gradient and a high-sample current-policy Monte Carlo score gradient are evaluated at the same (k, state, action). The robust controller receives only epsilon_i = ||g_MC - g_learned||; it never receives the error direction. This is an oracle-assisted, non-deployable test of the symmetric uncertainty-set formulation.", "",
        f"The run uses seeds {', '.join(map(str, protocol['seeds']))}, {protocol['actor_updates']} actor/TD updates and {protocol['reference_mc_episodes']} IID MC continuations per actor-batch endpoint. It reuses the matching Lean AM initialization and paired evaluation bank from `{protocol['paired_baseline_run']}`. The MC radius is a noisy point estimate, not a certified upper confidence bound.", "",
        "| Method | Success (%) | Collision (%) | Discounted return |", "|---|---:|---:|---:|"
    ]
    for method in ("learned", "oracle_radius_robust"):
        row = aggregate.loc[method]
        name = "Lean AM" if method == "learned" else "Oracle-radius robust QAM"
        lines.append(f"| {name} | {100*row.success_mean:.2f} ± {100*row.success_sd:.2f} | {100*row.collision_mean:.2f} ± {100*row.collision_sd:.2f} | {row.return_mean:.4f} ± {row.return_sd:.4f} |")
    lines.extend(["", f"Paired success difference (robust − Lean AM): **{result['mean_difference_pp']:+.2f} pp**; per-seed differences {', '.join(f'{value:+.2f}' for value in differences.values)} pp, {result['wins']} wins / {result['ties']} ties / {result['losses']} losses.", "",
        "Interpretation is intentionally narrow: improvement would show that knowing the error magnitude can help this symmetric robust update under this toy protocol, while failure would show that even privileged radii do not help in this setup. It would not show that a deployable model can estimate radii, nor establish an HJBI guarantee. The error direction is never given to the robust controller.", ""])
    (output / "report.md").write_text("\n".join(lines))


def run_radius(root, protocol):
    output = root / "oracle-radius"
    output.mkdir(parents=True, exist_ok=True)
    c = effective_config(protocol)
    payload = dict(protocol=protocol, qam_config=asdict(c),
        source_baseline_manifest_sha256=digest(ROOT / protocol["paired_baseline_run"] /
                                                "frozen-protocol.json"),
        source_code_sha256={name: digest(ROOT / name) for name in (
            "scripts/run_oracle_radius_gradient_diagnostic.py",
            "scripts/run_point_long_training.py",
            "scripts/point_reference_accelerated.py",
            "scripts/terminal_consistent_robust_qam.py",
            "notebooks/02_qam_point_obstacle.py",
            "notebooks/02_qam_td_terminal_error.py")})
    write_json(output / "frozen-protocol.json", payload)
    all_rows, all_pairs = [], []
    for seed in protocol["seeds"]:
        done = output / f"seed-{seed}" / "complete.json"
        if done.exists():
            saved = json.loads(done.read_text())
            all_rows.extend((saved["baseline"], saved["final"]))
            all_pairs.append(saved["paired"])
            continue
        baseline, final, paired = train_oracle_radius_seed(seed, c, protocol, output)
        baseline["method"] = "learned"
        all_rows.extend((baseline, final))
        all_pairs.append(paired)
    summarize_radius(all_rows, all_pairs, output, protocol)
    return output


def direction_decomposition(root, protocol):
    source = ROOT / protocol["direction_audit_source_run"]
    output = root / "direction-decomposition"
    output.mkdir(parents=True, exist_ok=True)
    point_config = json.loads((ROOT / "configs/reference_audit_points.json").read_text())
    category = {row["id"]: row["geometry_category"] for row in point_config["points"]}
    rows = []
    for seed in protocol["direction_audit_seeds"]:
        for update in protocol["direction_audit_checkpoints"]:
            path = source / f"seed-{seed}" / f"audit-learned-N{update}" / "gradients.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            for row in frame.to_dict("records"):
                g_l = torch.tensor([[row["critic_x"], row["critic_y"]]], dtype=torch.float64)
                g_r = torch.tensor([[row["mc_x"], row["mc_y"]]], dtype=torch.float64)
                se = np.hypot(row["se_x"], row["se_y"])
                dec = robust.decompose_gradient_errors(g_l, g_r)
                gl, gr = g_l.numpy()[0], g_r.numpy()[0]
                n_l, n_r = float(np.linalg.norm(gl)), float(np.linalg.norm(gr))
                reliable = n_r > max(.03, 3 * float(se))
                rel_learned = float(np.linalg.norm(gl - gr) / max(n_r, 1e-12))
                mag = dec["magnitude_only"].numpy()[0]
                direct = dec["direction_only"].numpy()[0]
                rows.append(dict(seed=seed, N=update, point_id=row["point_id"],
                    geometry_category=category.get(row["point_id"], "sampled current-policy endpoint"),
                    learned_norm=n_l, reference_norm=n_r, reference_se_norm=float(se),
                    magnitude_ratio=n_l / max(n_r, 1e-12),
                    cosine=float(dec["cosine"].item()),
                    angle_degrees=float(np.degrees(np.arccos(np.clip(
                        dec["cosine"].item(), -1., 1.)))) if n_l > 1e-12 and n_r > 1e-12 else np.nan,
                    reliable_reference_direction=bool(reliable),
                    learned_relative_error=rel_learned,
                    magnitude_only_relative_error=float(np.linalg.norm(mag-gr)/max(n_r, 1e-12)),
                    direction_only_relative_error=float(np.linalg.norm(direct-gr)/max(n_r, 1e-12)),
                    full_reference_relative_error=0.0))
    data = pd.DataFrame(rows)
    data.to_csv(output / "gradient-decomposition-points.csv", index=False)
    reliable = data[data.reliable_reference_direction & (data.learned_norm > 1e-12)].copy()
    if reliable.empty:
        raise RuntimeError("no reliable nondegenerate gradient pairs")

    def summarize(group):
        cosine = group.cosine.to_numpy()
        return pd.Series(dict(n=len(group),
            cosine_mean=float(np.mean(cosine)), cosine_median=float(np.median(cosine)),
            fraction_cosine_below_0=float(np.mean(cosine < 0.)),
            fraction_cosine_below_0_5=float(np.mean(cosine < .5)),
            angle_median_degrees=float(np.nanmedian(group.angle_degrees)),
            magnitude_ratio_median=float(np.median(group.magnitude_ratio)),
            learned_relative_error_mean=float(group.learned_relative_error.mean()),
            magnitude_only_relative_error_mean=float(group.magnitude_only_relative_error.mean()),
            direction_only_relative_error_mean=float(group.direction_only_relative_error.mean()),
            full_reference_relative_error_mean=0.))

    overall = summarize(reliable).to_dict()
    per_seed = reliable.groupby("seed", sort=True).apply(summarize, include_groups=False)
    per_seed.to_csv(output / "gradient-decomposition-by-seed.csv")
    per_checkpoint = reliable.groupby("N", sort=True).apply(summarize, include_groups=False)
    per_checkpoint.to_csv(output / "gradient-decomposition-by-checkpoint.csv")
    per_geometry = reliable.groupby("geometry_category", sort=True).apply(summarize,
                                                                          include_groups=False)
    per_geometry.to_csv(output / "gradient-decomposition-by-geometry.csv")
    write_json(output / "gradient-decomposition-summary.json", dict(
        source_run=protocol["direction_audit_source_run"],
        total_points=int(len(data)), reliable_nonzero_points=int(len(reliable)),
        direction_reliability_filter="||g_MC|| > max(0.03, 3 ||SE_MC||)",
        metrics=overall))

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in reliable.groupby("geometry_category"):
        ax.scatter(group.magnitude_ratio, group.cosine, s=25, alpha=.7, label=label)
    ax.axhline(0., color="black", linestyle="--", linewidth=.8)
    ax.axhline(.5, color="gray", linestyle=":", linewidth=.8)
    ax.set(xlabel="||g_learned|| / ||g_reference||", ylabel="gradient cosine",
           title="Held-out Q-gradient error: direction vs magnitude")
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1.), loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "gradient-direction-decomposition.png", dpi=180)
    plt.close(fig)
    one_step = one_step_decomposition(protocol, output)
    lines = ["# Read-only Q-gradient magnitude/direction decomposition", "",
        f"This analysis reuses the held-out current-policy MC gradient audits from `{protocol['direction_audit_source_run']}` for seeds {', '.join(map(str, protocol['direction_audit_seeds']))}. It includes {len(data)} paired critic/MC gradients over the declared checkpoints; {len(reliable)} nondegenerate points passed the predeclared direction-reliability filter `||g_MC|| > max(0.03, 3 ||SE_MC||)`. No model was retrained and no Oracle signal was used to update an Actor.", "",
        "For each point, `magnitude_only` keeps the learned direction and substitutes the MC gradient norm; `direction_only` keeps the learned norm and substitutes the MC direction. The full MC gradient is the privileged reference. Relative errors below are divided by `||g_MC||`.", "",
        "| Statistic | Value |", "|---|---:|",
        f"| Mean cosine | {overall['cosine_mean']:.3f} |",
        f"| Median cosine | {overall['cosine_median']:.3f} |",
        f"| Fraction cosine < 0 | {overall['fraction_cosine_below_0']:.1%} |",
        f"| Fraction cosine < 0.5 | {overall['fraction_cosine_below_0_5']:.1%} |",
        f"| Median learned/reference magnitude ratio | {overall['magnitude_ratio_median']:.3f} |",
        f"| Learned relative gradient error | {overall['learned_relative_error_mean']:.3f} |",
        f"| Magnitude-only relative error | {overall['magnitude_only_relative_error_mean']:.3f} |",
        f"| Direction-only relative error | {overall['direction_only_relative_error_mean']:.3f} |",
        "| Full-reference relative error | 0.000 (by construction) |", "",
        "This gradient-label decomposition is not by itself a causal performance comparison.", "",
        "![Gradient magnitude and direction decomposition](gradient-direction-decomposition.png)", "",
        "## One-step matched Actor intervention", "",
        f"At each paired Lean AM N=300 checkpoint, this diagnostic draws {protocol['one_step_endpoint_batch']} held-out states, uses the same generated Flow paths for each terminal-gradient variant, computes one virtual Adam update with the checkpoint's saved Adam history, and evaluates each resulting Actor on {protocol['one_step_evaluation_episodes']} common episodes. The MC reference uses {protocol['one_step_mc_episodes']} continuations per endpoint.", "",
        "| Terminal-gradient update | Mean success-rate change vs. no update (pp) |", "|---|---:|"]
    lines.extend(f"| {row['method']} | {row['mean_success_delta_pp']:+.2f} |" for row in one_step)
    lines.extend(["", "This is a one-step local mechanism probe, not a budget-matched training comparison. It can test whether a privileged correction improves the immediate update on these selected held-out states, but not whether repeated training maintains the gain.", ""])
    (output / "report.md").write_text("\n".join(lines))
    return output


def one_step_decomposition(protocol, output):
    """Compare privileged terminal-gradient substitutions from common snapshots."""
    rows = []
    methods = ("learned", "magnitude_only", "direction_only",
               "full_reference", "oracle_radius_robust")
    c = effective_config(protocol)
    task = M.PointObstacle(c)
    for seed in protocol["seeds"]:
        source = ROOT / protocol["paired_baseline_run"] / f"seed-{seed}"
        init = torch.load(source / "shared-initialization.pt", map_location="cpu",
                          weights_only=True)
        base = CORE.ConditionalVelocity(c)
        base.load_state_dict(init["base"])
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        saved = torch.load(source / "lean_am" / "lean_am-N300.pt",
                           map_location="cpu", weights_only=True)
        actor = CORE.ConditionalVelocity(c)
        actor.load_state_dict(saved["actor"])
        for parameter in actor.parameters():
            parameter.requires_grad_(True)
        critic = CORE.Critic(c)
        critic.load_state_dict(saved["critic"])
        optimizer = torch.optim.Adam(actor.parameters(), lr=c.actor_lr)
        optimizer.load_state_dict(saved["actor_optimizer"])
        archive = np.load(source / "valid-transitions.npz")
        valid = M.as_torch({key: archive[key] for key in archive.files})
        rng = CORE.generator(M.seed_for(seed, 980001))
        indices = torch.randperm(len(valid["k"]), generator=rng)[
            :protocol["one_step_endpoint_batch"]]
        k, y = valid["k"][indices], valid["y"][indices]
        z = torch.randn((len(indices), c.dimension), generator=rng)
        flow = torch.randn((len(indices), c.flow_steps, c.dimension), generator=rng)
        paths, _ = CORE.rollout(actor, base, c, k, y, z, flow)
        endpoint = paths[:, -1]
        g_learned = learned_gradient(critic, k, y, endpoint)
        g_reference, ref_se = mc_reference_batch(
            task, actor, base, c, seed, protocol["actor_updates"], k, y, endpoint,
            protocol["one_step_mc_episodes"], "cpu")
        pieces = robust.decompose_gradient_errors(g_learned, g_reference)
        epsilon = (g_reference - g_learned).norm(dim=-1)

        p_targets = {
            "learned": CORE.lean_adjoint(base, c, k, y, paths,
                                         -c.beta*g_learned),
            "magnitude_only": CORE.lean_adjoint(base, c, k, y, paths,
                                                -c.beta*pieces["magnitude_only"]),
            "direction_only": CORE.lean_adjoint(base, c, k, y, paths,
                                                -c.beta*pieces["direction_only"]),
            "full_reference": CORE.lean_adjoint(base, c, k, y, paths,
                                                -c.beta*g_reference),
        }
        p_targets["oracle_radius_robust"], _, robust_info = \
            robust.pathwise_terminal_robust_target(base, c, k, y, paths,
                                                   p_targets["learned"], epsilon)
        bank = M.noise_bank(c, protocol["one_step_evaluation_episodes"],
                            M.seed_for(seed, 502))
        before = M.evaluate(task, actor, base, bank)
        updated, parameter_grads = {}, {}
        for method in methods:
            q, loss = CORE.actor_gradient(actor, base, c, k, y, paths,
                                          p_targets[method])
            branch = copy.deepcopy(actor)
            branch_optimizer = torch.optim.Adam(branch.parameters(), lr=c.actor_lr)
            branch_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            CORE.set_gradient(branch, q)
            clipped = torch.nn.utils.clip_grad_norm_(branch.parameters(), c.actor_clip,
                                                    error_if_nonfinite=True)
            branch_optimizer.step()
            branch.zero_grad(set_to_none=True)
            updated[method] = branch
            parameter_grads[method] = q
            stats = M.evaluate(task, branch, base, bank)
            rows.append(dict(seed=seed, method=method, loss=loss,
                success_rate=stats["success_rate"],
                success_delta_pp=100*(stats["success_rate"]-before["success_rate"]),
                collision_rate=stats["collision_rate"],
                discounted_return=stats["discounted_return"],
                parameter_gradient_norm=float(q.norm()),
                pre_clip_norm=float(clipped),
                parameter_step_norm=float((CORE.parameter_vector(branch)-
                                           CORE.parameter_vector(actor)).norm()),
                reference_mc_se_rms=float(np.sqrt(np.mean(ref_se**2))),
                oracle_radius_mean=float(epsilon.mean()),
                robust_drift_ratio=robust_info["robust_to_nominal_drift_ratio"]))
        reference_parameter_grad = parameter_grads["full_reference"]
        for row in rows[-len(methods):]:
            row["matching_gradient_cosine_to_full_reference"] = float(
                torch.nn.functional.cosine_similarity(
                    parameter_grads[row["method"]], reference_parameter_grad, dim=0))
        rows.append(dict(seed=seed, method="no_update", loss=np.nan,
            success_rate=before["success_rate"], success_delta_pp=0.,
            collision_rate=before["collision_rate"],
            discounted_return=before["discounted_return"],
            parameter_gradient_norm=0., pre_clip_norm=0., parameter_step_norm=0.,
            reference_mc_se_rms=float(np.sqrt(np.mean(ref_se**2))),
            oracle_radius_mean=float(epsilon.mean()), robust_drift_ratio=1.,
            matching_gradient_cosine_to_full_reference=np.nan))
        print("ONE-STEP DIAGNOSTIC", seed, "epsilon", float(epsilon.mean()), flush=True)

    data = pd.DataFrame(rows)
    data.to_csv(output / "one-step-decomposition-by-seed.csv", index=False)
    summary = data[data.method != "no_update"].groupby("method").agg(
        mean_success_delta_pp=("success_delta_pp", "mean"),
        seed_sd_success_delta_pp=("success_delta_pp", "std"),
        mean_step_norm=("parameter_step_norm", "mean"),
        mean_step_cosine_to_full_ref=("matching_gradient_cosine_to_full_reference", "mean"))
    summary.to_csv(output / "one-step-decomposition-summary.csv")
    return [dict(method=method, mean_success_delta_pp=float(row.mean_success_delta_pp))
            for method, row in summary.iterrows()]


def create_root(path, protocol):
    if path is not None:
        if path.exists():
            metadata = path / "protocol.json"
            if not metadata.exists():
                raise FileExistsError(f"existing output has no protocol marker: {path}")
            saved = json.loads(metadata.read_text())
            if saved.get("config_sha256") != digest(PROTOCOL_PATH) \
                    or saved.get("protocol") != protocol:
                raise RuntimeError(f"existing output protocol differs: {path}")
            return path
        path.mkdir(parents=True, exist_ok=False)
        return path
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = ROOT / "runs" / f"02-qam-point-obstacle-oracle-radius-diagnostic-{stamp}-{secrets.token_hex(2)}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("radius", "direction", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    root = create_root(args.output, protocol)
    write_json(root / "protocol.json", dict(protocol=protocol,
        config_sha256=digest(PROTOCOL_PATH),
        base_protocol_sha256=digest(BASE_PROTOCOL_PATH)))
    existing_complete = root / "complete.json"
    previous = json.loads(existing_complete.read_text()).get("stages", {}) \
        if existing_complete.exists() else {}
    results = dict(previous)
    if args.stage in ("radius", "all"):
        results["oracle_radius"] = str(run_radius(root, protocol))
    if args.stage in ("direction", "all"):
        results["direction_decomposition"] = str(direction_decomposition(root, protocol))
    write_json(root / "complete.json", dict(stages=results, completed_at=time.time()))
    print("DIAGNOSTIC COMPLETE", json.dumps(dict(root=str(root), stages=results)), flush=True)


if __name__ == "__main__":
    main()
