"""Single-seed global scaling control for the PointObstacle QAM experiment.

The six global-scale branches only multiply each branch's learned terminal
Q-action gradient by a fixed c. They never call a reference-gradient estimator.
The prior privileged magnitude/full-reference results are read only after the
scale sweep and strict c=1 replay checks pass. A projection-only privileged
branch is run only if global scaling explains less than 80% of the magnitude
control's success-rate gain.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import secrets
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "point_global_gradient_scale_control.json"
BASELINE_DIR = ROOT / "reports" / "point_gradient_magnitude_direction"
EXISTING_RUNNER_PATH = ROOT / "scripts" / "run_point_gradient_magnitude_direction.py"

spec = importlib.util.spec_from_file_location("point_magnitude_direction_reference", EXISTING_RUNNER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot import existing PointObstacle runner: {EXISTING_RUNNER_PATH}")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)

M = experiment.M
CORE = experiment.CORE
torch.set_num_threads(4)

REFERENCE_ACCESS_CALLS = 0


class ReplayMismatch(RuntimeError):
    """The mandatory c=1 reproduction check failed; stop before interpretation."""


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scale_key(c: float) -> str:
    return f"c{float(c):.1f}"


def scale_gradient(learned_gradient: torch.Tensor, c: float) -> torch.Tensor:
    """Apply the only allowed global-scale intervention; preserve zero vectors."""
    if not math.isfinite(float(c)) or float(c) <= 0:
        raise ValueError("global gradient scale must be finite and positive")
    if not bool(torch.isfinite(learned_gradient).all()):
        raise FloatingPointError("non-finite learned gradient")
    return learned_gradient * float(c)


def projection_gradient(learned_gradient: torch.Tensor,
                        reference_gradient: torch.Tensor,
                        zero_tolerance: float = 1e-8) -> torch.Tensor:
    """Privileged projection: preserve the signed reference derivative along d_L."""
    if learned_gradient.shape != reference_gradient.shape or learned_gradient.ndim != 2:
        raise ValueError("gradients must have matching [batch, action_dim] shape")
    if not bool(torch.isfinite(learned_gradient).all()):
        raise FloatingPointError("non-finite learned gradient")
    if not bool(torch.isfinite(reference_gradient).all()):
        raise FloatingPointError("non-finite reference gradient")
    norm = learned_gradient.norm(dim=-1, keepdim=True)
    direction = torch.where(norm > zero_tolerance,
                            learned_gradient / norm.clamp_min(zero_tolerance),
                            torch.zeros_like(learned_gradient))
    signed_projection = (direction * reference_gradient).sum(dim=-1, keepdim=True)
    return signed_projection * direction


def _blocked_reference_estimator(*_args, **_kwargs):
    global REFERENCE_ACCESS_CALLS
    REFERENCE_ACCESS_CALLS += 1
    raise RuntimeError("reference-gradient access is forbidden during global-scale training")


@contextmanager
def block_reference_estimators():
    """Fail closed if global-scale training accidentally asks for an oracle label."""
    old_batch = experiment.mc_reference_batch
    old_independent = experiment.longrun.independent_mc
    experiment.mc_reference_batch = _blocked_reference_estimator
    experiment.longrun.independent_mc = _blocked_reference_estimator
    try:
        yield
    finally:
        experiment.mc_reference_batch = old_batch
        experiment.longrun.independent_mc = old_independent


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    if config.get("scales") != expected:
        raise RuntimeError(f"scales must remain frozen at {expected}")
    if (config.get("seed") != 59001 or config.get("stage1_actor_updates") != 100
            or config.get("beta") != 4.0):
        raise RuntimeError("this diagnostic is frozen to seed 59001 and N=100")
    return config


def scale_names(config):
    return [scale_key(value) for value in config["scales"]]


def assert_common_rng(states, names):
    first = states[names[0]]["rng"].get_state()
    for name in names[1:]:
        if not torch.equal(first, states[name]["rng"].get_state()):
            raise RuntimeError(f"matched training RNG streams diverged at {name}")


def checkpoint_payload(states, completed, config_hash, names):
    branches = {}
    for name in names:
        state = states[name]
        branches[name] = {
            key: state[key].state_dict() for key in ("actor", "critic", "target")
        }
        branches[name]["actor_optimizer"] = state["aopt"].state_dict()
        branches[name]["critic_optimizer"] = state["copt"].state_dict()
        branches[name]["rng"] = state["rng"].get_state()
    return {"completed_updates": int(completed), "config_hash": config_hash,
            "branch_names": names, "branches": branches}


def save_checkpoint(path, states, completed, config_hash, names):
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint_payload(states, completed, config_hash, names), tmp)
    tmp.replace(path)


def restore_checkpoint(path, states, config_hash, names):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["config_hash"] != config_hash or payload["branch_names"] != names:
        raise RuntimeError("checkpoint protocol mismatch; refusing to resume")
    for name in names:
        branch = payload["branches"][name]
        state = states[name]
        for key in ("actor", "critic", "target"):
            state[key].load_state_dict(branch[key])
        state["aopt"].load_state_dict(branch["actor_optimizer"])
        state["copt"].load_state_dict(branch["critic_optimizer"])
        state["rng"].set_state(branch["rng"])
    assert_common_rng(states, names)
    return int(payload["completed_updates"])


def initialize_run(config, output, device):
    c = experiment.load_effective_config(config)
    if not math.isclose(c.beta, config["beta"], rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(f"frozen beta mismatch: config={config['beta']} effective={c.beta}")
    if not math.isclose(c.actor_clip, config["actor_clip"], rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(f"frozen actor clip mismatch: config={config['actor_clip']} effective={c.actor_clip}")
    source, seed_dir, input_dir, _source_config, input_manifest = experiment.validate_source(c, config)
    init_path = input_dir / config["initialization_source"].split("/", 1)[1]
    train_path = input_dir / config["train_data_source"].split("/", 1)[1]
    valid_path = input_dir / config["valid_data_source"].split("/", 1)[1]
    init = torch.load(init_path, map_location="cpu", weights_only=True)
    base = CORE.ConditionalVelocity(c)
    base.load_state_dict(init["base"])
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    initial_critic = CORE.Critic(c)
    initial_critic.load_state_dict(init["critic"])
    initial_target = CORE.Critic(c)
    initial_target.load_state_dict(init["target"])
    train_np = np.load(train_path, allow_pickle=False)
    train = M.as_torch({key: train_np[key] for key in train_np.files})
    names = scale_names(config)
    states = {name: experiment.new_branch(base, initial_critic, initial_target,
                                          c, config["seed"])
              for name in names}
    reference_state = states[names[0]]
    for name in names[1:]:
        for field in ("actor", "critic", "target"):
            if M.model_hash(states[name][field]) != M.model_hash(reference_state[field]):
                raise RuntimeError(f"non-identical initial {field} in {name}")
        for field in ("aopt", "copt"):
            if states[name][field].state_dict() != reference_state[field].state_dict():
                raise RuntimeError(f"non-identical initial optimizer state in {name}")
    assert_common_rng(states, names)

    source_hashes = {
        "source_protocol": file_sha256(source / "protocol.json"),
        "source_complete": file_sha256(seed_dir / "complete.json"),
        "initialization_manifest": file_sha256(seed_dir / "initialization-source.json"),
        "shared_initialization": file_sha256(init_path),
        "train_transitions": file_sha256(train_path),
        "valid_transitions": file_sha256(valid_path),
        "long_training_runner": file_sha256(ROOT / "scripts/run_point_long_training.py"),
        "reference_runner": file_sha256(ROOT / "scripts/point_reference_accelerated.py"),
        "point_obstacle_notebook": file_sha256(ROOT / "notebooks/02_qam_point_obstacle.py"),
        "qam_core": file_sha256(ROOT / "notebooks/02_qam_td_terminal_error.py"),
        "existing_decomposition_runner": file_sha256(EXISTING_RUNNER_PATH),
        "global_scale_runner": file_sha256(Path(__file__)),
        "global_scale_config": file_sha256(CONFIG_PATH),
        "learned_only_replay_baseline": file_sha256(output / "replay-learned-baseline.csv"),
    }
    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device_for_reference_projection_only": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    frozen = {"protocol": config, "effective_config": asdict(c),
              "source_hashes": source_hashes, "runtime": runtime,
              "initial_branch_hashes": {
                  field: M.model_hash(reference_state[field])
                  for field in ("actor", "critic", "target")}}
    config_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text())
        if saved.get("config_hash") != config_hash:
            raise RuntimeError("output directory contains a different frozen protocol")
    else:
        write_json(protocol_path, {**frozen, "config_hash": config_hash,
                                   "created_at": time.time()})
    return c, source, seed_dir, train, base, initial_critic, initial_target, states, names, config_hash


def train_global_update(task, data, base, state, c, seed, update, scale,
                        train_logs):
    """One ordinary TD update and one Lean AM update with label c*g_L only."""
    actor, critic, target, rng = (state["actor"], state["critic"],
                                  state["target"], state["rng"])
    critic_ids = torch.randint(len(data["k"]), (c.critic_batch,), generator=rng)
    td_loss, q_mean = M.td_step(critic, target, state["copt"], data,
                                critic_ids, rng, c, actor, base)
    actor_ids = torch.randint(len(data["k"]), (c.actor_batch,), generator=rng)
    k, y = data["k"][actor_ids], data["y"][actor_ids]
    z = torch.randn((len(actor_ids), c.dimension), generator=rng)
    flow = torch.randn((len(actor_ids), c.flow_steps, c.dimension), generator=rng)
    paths, _ = CORE.rollout(actor, base, c, k, y, z, flow)
    learned = experiment.critic_action_gradient(critic, k, y, paths[:, -1])
    scaled = scale_gradient(learned, scale)
    terminal = -c.beta * scaled
    adjoints = CORE.lean_adjoint(base, c, k, y, paths, terminal)
    parameter_gradient, loss = CORE.actor_gradient(actor, base, c, k, y, paths, adjoints)
    CORE.set_gradient(actor, parameter_gradient)
    preclip = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), c.actor_clip,
                                                  error_if_nonfinite=True))
    postclip_sq = sum(float(parameter.grad.detach().square().sum())
                      for parameter in actor.parameters() if parameter.grad is not None)
    postclip = math.sqrt(postclip_sq)
    before = CORE.parameter_vector(actor).detach().clone()
    state["aopt"].step()
    after = CORE.parameter_vector(actor).detach().clone()
    actor_step_norm = float((after - before).norm())
    actor.zero_grad(set_to_none=True)
    row = {
        "seed": seed, "c": float(scale), "branch": scale_key(scale),
        "N": update + 1, "td_updates": update + 1, "actor_updates": update + 1,
        "td_loss": td_loss, "td_q_mean": q_mean, "matching_loss": loss,
        "learned_terminal_gradient_norm_mean": float(learned.norm(dim=-1).mean()),
        "scaled_terminal_gradient_norm_mean": float(scaled.norm(dim=-1).mean()),
        "terminal_adjoint_norm_mean": float(terminal.norm(dim=-1).mean()),
        "actor_parameter_gradient_norm": float(parameter_gradient.norm()),
        "actor_gradient_preclip_norm": preclip,
        "actor_gradient_postclip_norm": postclip,
        "clip_hit": bool(preclip > c.actor_clip),
        "actual_adam_step_norm": actor_step_norm,
    }
    train_logs.append(row)
    return row


def metric_rows(task, states, base, bank, names, N):
    rows = []
    for name in names:
        scale = float(name[1:])
        stats = M.evaluate(task, states[name]["actor"], base, bank)
        rows.append({"seed": 59001, "c": scale, "branch": name, "N": int(N),
                     "actor_updates": int(N), "td_updates": int(N), **stats})
    return rows


def compare_metrics(actual, expected, label, tolerance):
    metrics = ("success_rate", "collision_rate", "discounted_return")
    mismatches = {}
    for field in metrics:
        a, b = float(actual[field]), float(expected[field])
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance):
            mismatches[field] = {"actual": a, "expected": b, "difference": a - b}
    if mismatches:
        raise ReplayMismatch(f"{label} replay mismatch: {mismatches}")


def baseline_lookup(path, method=None, N=None, source=None):
    frame = pd.read_csv(path)
    chosen = frame
    if method is not None:
        chosen = chosen[chosen["method"] == method]
    if source is not None:
        chosen = chosen[chosen["source"] == source]
    if N is not None and "N" in chosen.columns:
        chosen = chosen[chosen["N"] == N]
    if len(chosen) != 1:
        raise RuntimeError(f"expected one baseline row for method={method}, source={source}, N={N} in {path}")
    return chosen.iloc[0].to_dict()


def write_learned_only_replay_baseline(output):
    """Extract only non-privileged Learned metrics before the global sweep."""
    rows = []
    with (BASELINE_DIR / "curves.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "learned":
                rows.append({"source": "curve", **{key: row[key] for key in
                              ("N", "success_rate", "collision_rate", "discounted_return", "episodes")}})
    if sorted(int(row["N"]) for row in rows) != [0, 10, 25, 50, 75, 100]:
        raise RuntimeError("replay baseline must contain exactly the six Learned checkpoints")
    final = []
    with (BASELINE_DIR / "final-independent.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "learned" and int(row["N"]) == 100:
                final.append({"source": "independent_final", **{key: row[key] for key in
                              ("N", "success_rate", "collision_rate", "discounted_return", "episodes")}})
    if len(final) != 1:
        raise RuntimeError("replay baseline must contain one Learned independent final result")
    path = output / "replay-learned-baseline.csv"
    if path.exists():
        saved = pd.read_csv(path)
        if len(saved) != 7 or set(saved["source"]) != {"curve", "independent_final"}:
            raise RuntimeError("existing Learned-only replay baseline is incomplete")
        return
    pd.DataFrame(rows + final).to_csv(path, index=False)


def replay_baseline_path(output):
    return output / "replay-learned-baseline.csv"


def record_checkpoint(task, states, base, bank, names, N, curves,
                      replay_rows, config, replay_path):
    rows = metric_rows(task, states, base, bank, names, N)
    curves = [row for row in curves if not (int(row["N"]) == N)] + rows
    control = next(row for row in rows if row["branch"] == "c1.0")
    if N == 0:
        for row in rows:
            compare_metrics(row, control, f"N=0 shared initialization {row['branch']}",
                            config["replay_tolerance"])
    if N in config["replay_updates"]:
        expected = baseline_lookup(replay_path, N=N, source="curve")
        compare_metrics(control, expected, f"c=1.0 vs prior Learned N={N}",
                        config["replay_tolerance"])
        replay_rows = [row for row in replay_rows if int(row["N"]) != N]
        replay_rows.append({
            "N": N, "episodes": int(control["episodes"]), "passed": True,
            "max_abs_metric_difference": max(
                abs(float(control[field]) - float(expected[field]))
                for field in ("success_rate", "collision_rate", "discounted_return")),
            "success_rate": control["success_rate"],
            "collision_rate": control["collision_rate"],
            "discounted_return": control["discounted_return"],
        })
    return curves, replay_rows, rows


def compute_global_results(final_rows, config):
    rows = sorted(final_rows, key=lambda row: float(row["c"]))
    c1 = next(row for row in rows if math.isclose(float(row["c"]), 1.0))
    # Ties are fixed to the smallest predeclared scale, without consulting any privileged control.
    best = min(rows, key=lambda row: (-float(row["success_rate"]), float(row["c"])))
    prior = pd.read_csv(BASELINE_DIR / "final-independent.csv")
    mag = float(prior.loc[prior["method"] == "magnitude_only", "success_rate"].iloc[0])
    full = float(prior.loc[prior["method"] == "full_reference", "success_rate"].iloc[0])
    learned = float(c1["success_rate"])
    mag_den = mag - learned
    full_den = full - learned
    recovery_mag = (float(best["success_rate"]) - learned) / mag_den
    recovery_full = (float(best["success_rate"]) - learned) / full_den
    if recovery_mag >= 0.80:
        case = "A"
    elif recovery_mag <= 0.30:
        case = "C"
    else:
        case = "B"
    result = []
    for row in rows:
        score = float(row["success_rate"])
        result.append({
            "c": float(row["c"]), "effective_beta": float(row["c"]) * config["beta"],
            "success_percent": 100.0 * score,
            "collision_percent": 100.0 * float(row["collision_rate"]),
            "discounted_return": float(row["discounted_return"]),
            "delta_vs_c1_pp": 100.0 * (score - learned),
            "recovery_of_magnitude_only": (score - learned) / mag_den,
            "recovery_of_full_reference": (score - learned) / full_den,
            "episodes": int(row["episodes"]),
        })
    summary = {
        "best_global_scale": float(best["c"]),
        "best_global_success_percent": 100.0 * float(best["success_rate"]),
        "learned_success_percent": 100.0 * learned,
        "prior_magnitude_only_success_percent": 100.0 * mag,
        "prior_full_reference_success_percent": 100.0 * full,
        "recovery_of_magnitude_only": recovery_mag,
        "recovery_of_full_reference": recovery_full,
        "case": case,
        "projection_required": case in ("B", "C"),
    }
    return result, summary


def update_diagnostics(task, states, base, c, diag, N, training_logs, names):
    rows = []
    k, y, z, flow = diag["k"], diag["y"], diag["z"], diag["flow"]
    for name in names:
        scale = float(name[1:])
        state = states[name]
        paths, _ = CORE.rollout(state["actor"], base, c, k, y, z, flow)
        learned = experiment.critic_action_gradient(state["critic"], k, y, paths[:, -1])
        scaled = scale_gradient(learned, scale)
        terminal = -c.beta * scaled
        adjoints = CORE.lean_adjoint(base, c, k, y, paths, terminal)
        parameter_gradient, loss = CORE.actor_gradient(state["actor"], base, c, k, y,
                                                       paths, adjoints)
        CORE.set_gradient(state["actor"], parameter_gradient)
        preclip = float(torch.nn.utils.clip_grad_norm_(state["actor"].parameters(),
                                                      c.actor_clip, error_if_nonfinite=True))
        postclip_sq = sum(float(parameter.grad.detach().square().sum())
                          for parameter in state["actor"].parameters()
                          if parameter.grad is not None)
        postclip = math.sqrt(postclip_sq)
        state["actor"].zero_grad(set_to_none=True)
        virtual_step = CORE.virtual_adam(state["actor"], state["aopt"],
                                         parameter_gradient, c)
        history = [row for row in training_logs
                   if row["branch"] == name and int(row["N"]) <= int(N)]
        rows.append({
            "N": int(N), "c": scale, "branch": name, "samples": int(len(k)),
            "terminal_gradient_norm": float(scaled.norm(dim=-1).mean()),
            "terminal_adjoint_norm": float(terminal.norm(dim=-1).mean()),
            "matching_parameter_gradient_norm": float(parameter_gradient.norm()),
            "pre_clip_parameter_gradient_norm": preclip,
            "post_clip_parameter_gradient_norm": postclip,
            "fixed_batch_clip_hit": bool(preclip > c.actor_clip),
            "training_clip_fraction": (float(np.mean([r["clip_hit"] for r in history]))
                                       if history else None),
            "virtual_adam_step_norm": float(virtual_step.norm()),
            "mean_actual_adam_step_norm": (float(np.mean([r["actual_adam_step_norm"] for r in history]))
                                           if history else None),
            "matching_loss": float(loss),
        })
    learned = next(row for row in rows if row["branch"] == "c1.0")
    for row in rows:
        row["terminal_gradient_norm_ratio_vs_c1"] = (
            row["terminal_gradient_norm"] / (learned["terminal_gradient_norm"] + 1e-15))
        row["parameter_gradient_norm_ratio_vs_c1"] = (
            row["matching_parameter_gradient_norm"] /
            (learned["matching_parameter_gradient_norm"] + 1e-15))
        row["virtual_adam_step_norm_ratio_vs_c1"] = (
            row["virtual_adam_step_norm"] / (learned["virtual_adam_step_norm"] + 1e-15))
    return rows


def prior_update_diagnostics(current_rows, output_path):
    """Append previous privileged controls only after global-scale training ends."""
    source = pd.read_csv(BASELINE_DIR / "actor_update_decomposition.csv")
    selected = source[source["method"].isin(["learned", "magnitude_only", "full_reference"])]
    global_rows = pd.DataFrame(current_rows)
    output = list(current_rows)
    learned_ok = True
    for N in (0, 50, 100):
        current = global_rows[(global_rows["N"] == N) & (global_rows["branch"] == "c1.0")].iloc[0]
        previous = selected[(selected["N"] == N) & (selected["method"] == "learned")].iloc[0]
        learned_ok &= math.isclose(float(current["matching_parameter_gradient_norm"]),
                                   float(previous["parameter_gradient_norm"]),
                                   rel_tol=0, abs_tol=1e-5)
        learned_ok &= math.isclose(float(current["virtual_adam_step_norm"]),
                                   float(previous["virtual_adam_step_norm"]),
                                   rel_tol=0, abs_tol=1e-5)
    if learned_ok:
        for _, previous in selected[selected["method"].isin(["magnitude_only", "full_reference"])].iterrows():
            output.append({
                "N": int(previous["N"]), "c": None,
                "branch": f"prior_{previous['method']}", "samples": int(previous["samples"]),
                "terminal_gradient_norm": None, "terminal_adjoint_norm": None,
                "matching_parameter_gradient_norm": float(previous["parameter_gradient_norm"]),
                "pre_clip_parameter_gradient_norm": None,
                "post_clip_parameter_gradient_norm": None,
                "fixed_batch_clip_hit": None, "training_clip_fraction": None,
                "virtual_adam_step_norm": float(previous["virtual_adam_step_norm"]),
                "mean_actual_adam_step_norm": None, "matching_loss": float(previous["matching_loss"]),
                "terminal_gradient_norm_ratio_vs_c1": None,
                "parameter_gradient_norm_ratio_vs_c1": None,
                "virtual_adam_step_norm_ratio_vs_c1": None,
            })
    pd.DataFrame(output).sort_values(["N", "branch"]).to_csv(output_path, index=False)
    return bool(learned_ok)


def plot_global_scale(curves, results, prior, output):
    result_frame = pd.DataFrame(results).sort_values("c")
    mag = float(prior.loc[prior["method"] == "magnitude_only", "success_rate"].iloc[0]) * 100
    full = float(prior.loc[prior["method"] == "full_reference", "success_rate"].iloc[0]) * 100
    learned = float(prior.loc[prior["method"] == "learned", "success_rate"].iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(result_frame["c"], result_frame["success_percent"], marker="o", linewidth=2,
            label="Global scale (N=100)")
    ax.axhline(learned, color="black", linestyle="--", label="Learned c=1.0")
    ax.axhline(mag, color="#2878b5", linestyle=":", label="Prior magnitude-only")
    ax.axhline(full, color="#d95f02", linestyle="-.", label="Prior full-reference")
    ax.set(xlabel="Global terminal-gradient scale c", ylabel="N=100 success rate (%)",
           xticks=result_frame["c"], ylim=(0, 100))
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "success-vs-global-scale.png", dpi=180)
    plt.close(fig)

    curve_frame = pd.DataFrame(curves).sort_values(["c", "N"])
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for c_value, group in curve_frame.groupby("c"):
        ax.plot(group["N"], 100 * group["success_rate"], marker="o", linewidth=1.7,
                label=f"c={float(c_value):g}")
    ax.axhline(mag, color="#2878b5", linestyle=":", label="Prior magnitude-only N=100")
    ax.axhline(full, color="#d95f02", linestyle="-.", label="Prior full-reference N=100")
    ax.set(xlabel="Actor / TD updates N", ylabel="Success rate (%)",
           xticks=sorted(curve_frame["N"].unique()), ylim=(0, 100))
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "success-curves-by-scale.png", dpi=180)
    plt.close(fig)


def train_projection(task, data, base, initial_critic, initial_target, c, config,
                     output, device, curve_bank, final_bank, diag):
    """Conditional privileged projection branch; called only for recovery < 0.80."""
    name = "projection_only"
    state = experiment.new_branch(base, initial_critic, initial_target, c, config["seed"])
    rng = state["rng"]
    curve_rows, train_rows, diag_rows = [], [], []
    selected = set(config["stage1_curve_updates"])
    diag_steps = set(config["actor_update_diagnostic_updates"])
    mc_calls_before = REFERENCE_ACCESS_CALLS
    for N in range(0, config["stage1_actor_updates"] + 1):
        if N in selected:
            stats = M.evaluate(task, state["actor"], base, curve_bank)
            curve_rows.append({"seed": 59001, "branch": name, "N": N,
                               "actor_updates": N, "td_updates": N, **stats})
        if N in diag_steps:
            # Metrics are computed on a fixed diagnostic batch under the actual current policy.
            k, y, z, flow = diag["k"], diag["y"], diag["z"], diag["flow"]
            paths, _ = CORE.rollout(state["actor"], base, c, k, y, z, flow)
            learned = experiment.critic_action_gradient(state["critic"], k, y, paths[:, -1])
            reference, _se, _q, _qse, seconds = experiment.mc_reference_batch(
                task, state["actor"], base, c, config["seed"], N, k, y, paths[:, -1],
                config["reference_mc_episodes"], device, role=720000)
            projected = projection_gradient(learned, reference,
                                            config["gradient_zero_tolerance"])
            terminal = -c.beta * projected
            adjoints = CORE.lean_adjoint(base, c, k, y, paths, terminal)
            q, loss = CORE.actor_gradient(state["actor"], base, c, k, y, paths, adjoints)
            virtual = CORE.virtual_adam(state["actor"], state["aopt"], q, c)
            diag_rows.append({"N": N, "branch": name, "samples": len(k),
                              "terminal_gradient_norm": float(projected.norm(dim=-1).mean()),
                              "terminal_adjoint_norm": float(terminal.norm(dim=-1).mean()),
                              "matching_parameter_gradient_norm": float(q.norm()),
                              "virtual_adam_step_norm": float(virtual.norm()),
                              "matching_loss": float(loss), "mc_seconds": seconds})
        if N == config["stage1_actor_updates"]:
            break
        critic_ids = torch.randint(len(data["k"]), (c.critic_batch,), generator=rng)
        td_loss, q_mean = M.td_step(state["critic"], state["target"], state["copt"],
                                    data, critic_ids, rng, c, state["actor"], base)
        actor_ids = torch.randint(len(data["k"]), (c.actor_batch,), generator=rng)
        k, y = data["k"][actor_ids], data["y"][actor_ids]
        z = torch.randn((len(actor_ids), c.dimension), generator=rng)
        flow = torch.randn((len(actor_ids), c.flow_steps, c.dimension), generator=rng)
        paths, _ = CORE.rollout(state["actor"], base, c, k, y, z, flow)
        learned = experiment.critic_action_gradient(state["critic"], k, y, paths[:, -1])
        reference, _se, _q, _qse, seconds = experiment.mc_reference_batch(
            task, state["actor"], base, c, config["seed"], N, k, y, paths[:, -1],
            config["reference_mc_episodes"], device, role=710000)
        projected = projection_gradient(learned, reference,
                                        config["gradient_zero_tolerance"])
        terminal = -c.beta * projected
        adjoints = CORE.lean_adjoint(base, c, k, y, paths, terminal)
        q, loss = CORE.actor_gradient(state["actor"], base, c, k, y, paths, adjoints)
        CORE.set_gradient(state["actor"], q)
        preclip = float(torch.nn.utils.clip_grad_norm_(state["actor"].parameters(),
                                                      c.actor_clip, error_if_nonfinite=True))
        state["aopt"].step()
        state["actor"].zero_grad(set_to_none=True)
        train_rows.append({"N": N + 1, "td_updates": N + 1, "actor_updates": N + 1,
                           "td_loss": td_loss, "td_q_mean": q_mean,
                           "matching_loss": loss, "pre_clip_norm": preclip,
                           "clip_hit": bool(preclip > c.actor_clip),
                           "reference_mc_seconds": seconds,
                           "learned_gradient_norm_mean": float(learned.norm(dim=-1).mean()),
                           "projected_gradient_norm_mean": float(projected.norm(dim=-1).mean())})
        if (N + 1) % 10 == 0:
            print("PROJECTION UPDATE", N + 1, "of", config["stage1_actor_updates"], flush=True)
    if REFERENCE_ACCESS_CALLS != mc_calls_before:
        raise RuntimeError("projection run unexpectedly invoked the global-scale fail-closed guard")
    final_stats = M.evaluate(task, state["actor"], base, final_bank)
    projection_final = [{"seed": 59001, "N": 100, "branch": name, **final_stats}]
    pd.DataFrame(curve_rows).to_csv(output / "projection-curves.csv", index=False)
    pd.DataFrame(projection_final).to_csv(output / "projection-final-independent.csv", index=False)
    pd.DataFrame(train_rows).to_csv(output / "projection-training-log.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(output / "projection-update-diagnostics.csv", index=False)
    return projection_final[0], curve_rows, train_rows, diag_rows


def write_report(output, global_results, summary, curves, diagnostics,
                 prior_compatible, update_strength, projection_result=None,
                 projection_curves=None):
    rows = [
        "# Global Q-gradient scale control: single-seed diagnostic", "",
        "## Direct answers", "",
        f"1. Best predeclared fixed global scale: **c={summary['best_global_scale']:g}** "
        f"(ties resolve to the smaller c).",
        f"2. Its independent N=100 success rate: **{summary['best_global_success_percent']:.4f}%**.",
        f"3. Recovery of prior magnitude-only benefit: "
        f"**{100*summary['recovery_of_magnitude_only']:.2f}%**.",
        f"4. Recovery of the prior full-reference gap: "
        f"**{100*summary['recovery_of_full_reference']:.2f}%**.",
        f"5. Predeclared interpretation: **Case {summary['case']}**.",
        f"6. At N=100 the best c versus c=1 has terminal-gradient norm ratio "
        f"{update_strength['terminal_gradient_ratio']:.3f}, matching-parameter-gradient ratio "
        f"{update_strength['parameter_gradient_ratio']:.3f}, virtual Adam-step ratio "
        f"{update_strength['virtual_adam_ratio']:.3f}, mean actual Adam-step ratio "
        f"{update_strength['actual_adam_ratio']:.3f}, and training clip fractions "
        f"{update_strength['clip_fraction_best']:.1%} vs {update_strength['clip_fraction_c1']:.1%}. "
        f"Interpretation: {update_strength['interpretation']}.", "",
        "The result is a single-seed mechanism diagnostic, not a deployable algorithm or a "
        "general claim. The scale sweep stopped at N=100; no N=300 training or extra c values were run.", "",
        "## N=100 global-scale results", "",
        "| c | effective beta | Success % | Collision % | Discounted return | Δ vs c=1 (pp) | "
        "Recovery of magnitude-only |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_results:
        rows.append(f"| {row['c']:.1f} | {row['effective_beta']:.1f} | "
                    f"{row['success_percent']:.4f} | {row['collision_percent']:.4f} | "
                    f"{row['discounted_return']:.8f} | {row['delta_vs_c1_pp']:+.4f} | "
                    f"{100*row['recovery_of_magnitude_only']:.2f}% |")
    rows += ["", "Prior privileged controls, not branches of this global sweep:", "",
             f"- Magnitude-only reference: **{summary['prior_magnitude_only_success_percent']:.4f}%**.",
             f"- Full reference: **{summary['prior_full_reference_success_percent']:.4f}%**.", "",
             "![Success rate versus fixed global scale](success-vs-global-scale.png)", "",
             "![Unsmooth success curves at the fixed checkpoints](success-curves-by-scale.png)", "",
             "## c=1 replay and matched design", "",
             "At N=0 all six actors are identical. The c=1 branch was compared against the "
             "published Learned branch at every fixed checkpoint (N=0,10,25,50,75,100) and on "
             "the independent 4096-episode N=100 evaluation. The run proceeds to interpretation "
             "only if success, collision, and discounted return match to the frozen 1e-12 absolute tolerance.", "",
             "The six branches share the exact seed-59001 initialization, TD/Actor RNG streams, "
             "minibatch draws, Flow noise, environment and evaluation banks. Each branch has its "
             "own Actor and Critic trajectory after updates. beta remains 4; only the learned "
             "terminal Q-gradient is multiplied by c.", "",
             "## Update-scale and clipping diagnostics", "",
             f"The previous privileged update diagnostics are protocol-compatible with the c=1 "
             f"replay: **{prior_compatible}**. Rows for prior magnitude-only/full-reference "
             "are copied into `actor-update-scale-diagnostics.csv` only after the global training "
             "and c=1 checks pass.", "",
             "For the best c, inspect the fixed-batch terminal-gradient, terminal-adjoint, "
             "matching-gradient, pre/post-clip norms and virtual Adam-step norm ratios versus c=1 "
             "at N=0/50/100. `training_clip_fraction` and `mean_actual_adam_step_norm` summarize "
             "actual updates through each checkpoint. Clip threshold is 5.0; terminal-gradient "
             "scale is not equivalent to an equal Actor-step scale under Adam/clipping.", "",
             "## Conditional projection diagnostic", ""]
    if summary["case"] == "A":
        rows += ["Case A passed (at least 80% of magnitude-only recovery). Per the frozen rule, "
                 "the projection-only privileged experiment was **not run**; no adaptive or "
                 "state-dependent calibration is pursued from this seed.", ""]
    elif projection_result is not None:
        current_final = pd.read_csv(output / "final-independent.csv")
        best_global = current_final[
            np.isclose(current_final["c"].astype(float), summary["best_global_scale"])
        ].iloc[0].to_dict()
        comparison = [
            ("Learned (c=1)", baseline_lookup(BASELINE_DIR / "final-independent.csv", "learned", 100)),
            (f"Best global c={summary['best_global_scale']:g}", best_global),
            ("Magnitude-only reference", baseline_lookup(
                BASELINE_DIR / "final-independent.csv", "magnitude_only", 100)),
            ("Full reference", baseline_lookup(
                BASELINE_DIR / "final-independent.csv", "full_reference", 100)),
            ("Projection-only (privileged)", projection_result),
        ]
        rows += ["Global scaling was insufficient under the predeclared Case B/C rule, so one "
                 "privileged projection-only branch was run from the same seed-59001 initialization "
                 "for 100 updates. It used the current branch's current-policy MC gradient only "
                 "for its Actor terminal labels; no reference information entered TD or Critic training.", "",
                 f"Projection-only success: **{100*float(projection_result['success_rate']):.4f}%** "
                 f"(collision {100*float(projection_result['collision_rate']):.4f}%, "
                 f"discounted return {float(projection_result['discounted_return']):.8f}).", "",
                 "| Comparison branch | Success % | Collision % | Discounted return |",
                 "|---|---:|---:|---:|"]
        for label, row in comparison:
            rows.append(f"| {label} | {100*float(row['success_rate']):.4f} | "
                        f"{100*float(row['collision_rate']):.4f} | "
                        f"{float(row['discounted_return']):.8f} |")
        rows += ["",
                 "This is a privileged directional-derivative diagnostic, not a deployable estimator. "
                 "It retains the sign of the reference gradient projected onto the learned direction.", ""]
    else:
        rows += ["Global scaling was insufficient under the predeclared Case B/C rule, but the "
                 "conditional projection run did not complete; see `failure.json`. No mechanistic "
                 "claim about projection is made.", ""]
    rows += ["## Evidence boundary", "",
             "Only seed 59001 was used. c* is the highest-success member of the six fixed global "
             "scales and is reported as a diagnostic, not selected as a production hyperparameter. "
             "The prior privileged scores are used only for post-training gap-recovery calculations; "
             "they did not select c. A Case A result means that on this seed much of the apparent "
             "magnitude-only gain is confounded with ordinary QAM guidance-strength calibration. "
             "A Case B/C result only motivates further sample-dependent analysis; it does not prove "
             "such calibration is necessary.", "",
             "The global training/evaluation phase had fail-closed guards around both the current "
             "runner's MC-reference estimator and its underlying independent-MC gradient estimator. "
             "`complete.json` records zero reference-gradient calls in that phase. No HJBI ball, "
             "ensemble, covariance, SVD, new Critic, new reward/environment, adaptive c, extra c, "
             "or best-checkpoint selection was used.", "",
             "## Artifacts", "",
             "- `curves.csv`: common-noise 1024-episode evaluation at N=0/10/25/50/75/100.",
             "- `final-independent.csv`: common-noise 4096-episode N=100 final evaluation.",
             "- `scale-results.csv`: requested main table and predeclared recovery calculations.",
             "- `actor-update-scale-diagnostics.csv`: fixed-batch gradient, clipping, virtual and actual step-scale diagnostics.",
             "- `protocol.json`, `complete.json`: frozen settings, source hashes, and completion/replay evidence.",
             "- `training-log.csv`: per-update losses, gradient norms, clip hits, and actual Adam-step norms.",
             "- `success-vs-global-scale.png`, `success-curves-by-scale.png`: unsmoothed plots.",
             ""]
    (output / "report.md").write_text("\n".join(rows))


def update_strength_summary(diagnostics, best_c):
    frame = pd.DataFrame(diagnostics)
    c1 = frame[(frame["N"] == 100) & (frame["branch"] == "c1.0")].iloc[0]
    best = frame[(frame["N"] == 100) &
                 np.isclose(frame["c"].astype(float), float(best_c))].iloc[0]
    actual_ratio = float(best["mean_actual_adam_step_norm"] /
                         max(float(c1["mean_actual_adam_step_norm"]), 1e-15))
    clip_best = float(best["training_clip_fraction"])
    clip_c1 = float(c1["training_clip_fraction"])
    virtual_ratio = float(best["virtual_adam_step_norm_ratio_vs_c1"])
    parameter_ratio = float(best["parameter_gradient_norm_ratio_vs_c1"])
    terminal_ratio = float(best["terminal_gradient_norm_ratio_vs_c1"])
    if clip_best - clip_c1 >= 0.10:
        interpretation = "the best scale triggers materially more clipping, so clipping interaction is a plausible contributor"
    elif actual_ratio >= 1.10 and virtual_ratio >= 1.10:
        interpretation = "larger virtual and realized Adam steps accompany the gain"
    elif parameter_ratio >= 1.10 and actual_ratio < 1.10:
        interpretation = "raw matching gradients are larger, but Adam largely cancels that scale in realized steps"
    elif actual_ratio < 1.10 and virtual_ratio < 1.10:
        interpretation = "the measured Adam-step magnitude changes little, so success differences are not explained by a simple larger-step norm"
    else:
        interpretation = "the update-strength and clipping diagnostics show a mixed effect; inspect the full per-checkpoint table"
    return {
        "terminal_gradient_ratio": terminal_ratio,
        "parameter_gradient_ratio": parameter_ratio,
        "virtual_adam_ratio": virtual_ratio,
        "actual_adam_ratio": actual_ratio,
        "clip_fraction_best": clip_best,
        "clip_fraction_c1": clip_c1,
        "interpretation": interpretation,
    }


def make_plots(output, curves, global_results):
    prior = pd.read_csv(BASELINE_DIR / "final-independent.csv")
    plot_global_scale(curves, global_results, prior, output)


def run(output: Path, device=None):
    global REFERENCE_ACCESS_CALLS
    config = load_config()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but unavailable")
    if output.exists():
        if not (output / "protocol.json").exists():
            raise FileExistsError(f"refusing an existing un-frozen output directory: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    write_learned_only_replay_baseline(output)
    (output / "checkpoints").mkdir(exist_ok=True)
    c, source, source_seed_dir, train, base, initial_critic, initial_target, states, names, config_hash = (
        initialize_run(config, output, device))
    task = M.PointObstacle(c)
    diag = experiment.diagnostic_batch(train, c, config["seed"])
    curve_bank = M.noise_bank(c, config["curve_episodes"], M.seed_for(config["seed"], 501))
    final_bank = M.noise_bank(c, config["final_episodes"], M.seed_for(config["seed"], 502))
    checkpoint_path = output / "suite-latest.pt"
    completed = restore_checkpoint(checkpoint_path, states, config_hash, names) if checkpoint_path.exists() else 0
    if completed > config["stage1_actor_updates"]:
        raise RuntimeError("checkpoint exceeds frozen N=100 target")
    curves = experiment.load_rows(output / "curves.csv")
    replay_rows = experiment.load_rows(output / "replay-sanity.csv")
    training_logs = experiment.load_rows(output / "training-log.csv")
    diagnostics = experiment.load_rows(output / "actor-update-scale-diagnostics.csv")
    if completed == 0 and not any(int(row["N"]) == 0 for row in curves):
        with block_reference_estimators():
            curves, replay_rows, _rows = record_checkpoint(
                task, states, base, curve_bank, names, 0, curves, replay_rows, config,
                replay_baseline_path(output))
            pd.DataFrame(curves).sort_values(["N", "c"]).to_csv(output / "curves.csv", index=False)
            pd.DataFrame(replay_rows).sort_values("N").to_csv(output / "replay-sanity.csv", index=False)
    if completed == 0 and not diagnostics:
        with block_reference_estimators():
            diagnostics.extend(update_diagnostics(task, states, base, c, diag, 0,
                                                  training_logs, names))
            pd.DataFrame(diagnostics).to_csv(output / "actor-update-scale-diagnostics.csv", index=False)

    # No global-scale branch may obtain an oracle/MC reference gradient. This
    # guard covers training, all fixed evaluation checkpoints, and final replay.
    REFERENCE_ACCESS_CALLS = 0
    with block_reference_estimators():
        for update in range(completed, config["stage1_actor_updates"]):
            for scale, name in zip(config["scales"], names):
                train_global_update(task, train, base, states[name], c,
                                    config["seed"], update, scale, training_logs)
            assert_common_rng(states, names)
            N = update + 1
            if N in config["stage1_curve_updates"]:
                curves, replay_rows, _rows = record_checkpoint(
                    task, states, base, curve_bank, names, N, curves, replay_rows, config,
                    replay_baseline_path(output))
                pd.DataFrame(curves).sort_values(["N", "c"]).to_csv(output / "curves.csv", index=False)
                pd.DataFrame(replay_rows).sort_values("N").to_csv(output / "replay-sanity.csv", index=False)
            if N in config["actor_update_diagnostic_updates"]:
                diagnostics.extend(update_diagnostics(task, states, base, c, diag, N,
                                                      training_logs, names))
                pd.DataFrame(diagnostics).to_csv(
                    output / "actor-update-scale-diagnostics.csv", index=False)
            if N % 10 == 0:
                pd.DataFrame(training_logs).to_csv(output / "training-log.csv", index=False)
                save_checkpoint(checkpoint_path, states, N, config_hash, names)
                write_json(output / "progress.json", {
                    "seed": config["seed"], "completed_actor_updates": N,
                    "completed_td_updates": N, "target_updates": 100,
                    "updated_at": time.time()})
                print("GLOBAL SCALE UPDATE", N, "of 100", flush=True)
        final_rows = []
        for name in names:
            final_rows.append({"seed": 59001, "N": 100, "c": float(name[1:]),
                               "branch": name,
                               **M.evaluate(task, states[name]["actor"], base, final_bank)})
        # The final c=1 replay is a hard stop, before any privileged score is read.
        expected_final = baseline_lookup(replay_baseline_path(output), N=100,
                                         source="independent_final")
        control_final = next(row for row in final_rows if row["branch"] == "c1.0")
        compare_metrics(control_final, expected_final, "c=1.0 vs prior Learned final N=100",
                        config["replay_tolerance"])
        replay_rows = [row for row in replay_rows if int(row["N"]) != 100]
        replay_rows.append({"N": 100, "episodes": int(control_final["episodes"]),
                            "passed": True,
                            "max_abs_metric_difference": max(
                                abs(float(control_final[field]) - float(expected_final[field]))
                                for field in ("success_rate", "collision_rate", "discounted_return")),
                            "success_rate": control_final["success_rate"],
                            "collision_rate": control_final["collision_rate"],
                            "discounted_return": control_final["discounted_return"]})
        pd.DataFrame(replay_rows).sort_values("N").to_csv(output / "replay-sanity.csv", index=False)
        pd.DataFrame(final_rows).sort_values("c").to_csv(output / "final-independent.csv", index=False)
        if REFERENCE_ACCESS_CALLS != 0:
            raise RuntimeError(f"unexpected reference-gradient calls: {REFERENCE_ACCESS_CALLS}")
    pd.DataFrame(training_logs).to_csv(output / "training-log.csv", index=False)
    global_results, summary = compute_global_results(final_rows, config)
    pd.DataFrame(global_results).sort_values("c").to_csv(output / "scale-results.csv", index=False)
    prior_compatible = prior_update_diagnostics(diagnostics, output / "actor-update-scale-diagnostics.csv")
    update_strength = update_strength_summary(diagnostics, summary["best_global_scale"])
    pd.DataFrame(curves).sort_values(["N", "c"]).to_csv(output / "curves.csv", index=False)
    make_plots(output, curves, global_results)

    projection_result = None
    projection_curves = None
    projection_status = "not_required_case_a"
    if summary["projection_required"]:
        print("GLOBAL SCALE RECOVERY BELOW 0.80; running conditional projection-only diagnostic", flush=True)
        projection_result, projection_curves, _projection_logs, _projection_diag = train_projection(
            task, train, base, initial_critic, initial_target, c, config, output,
            device, curve_bank, final_bank, diag)
        projection_status = "complete"
        pd.DataFrame([{**row, "comparison": "projection_only"}
                      for row in [projection_result]]).to_csv(
            output / "projection-final-independent.csv", index=False)
    write_report(output, global_results, summary, curves, diagnostics,
                 prior_compatible, update_strength, projection_result, projection_curves)
    write_json(output / "complete.json", {
        "seed": 59001, "actor_updates": 100, "td_updates": 100,
        "global_scales": config["scales"], "global_reference_gradient_calls": REFERENCE_ACCESS_CALLS,
        "c1_replay_passed": True, "c1_replay_updates": config["replay_updates"],
        "prior_update_diagnostics_protocol_compatible": bool(prior_compatible),
        "best_global_scale": summary["best_global_scale"],
        "global_recovery_of_magnitude_only": summary["recovery_of_magnitude_only"],
        "interpretation_case": summary["case"], "projection_status": projection_status,
        "completed_at": time.time(), "status": "complete"})
    print("GLOBAL SCALE DIAGNOSTIC COMPLETE", output,
          json.dumps(summary, sort_keys=True), flush=True)


def make_unique_output():
    return ROOT / "runs" / f"02-qam-point-global-gradient-scale-59001-{secrets.token_hex(4)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None,
                        help="device for the conditional privileged MC projection only")
    args = parser.parse_args()
    output = args.output.resolve() if args.output else make_unique_output()
    try:
        run(output, args.device)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", {"error": repr(exc),
                   "traceback": __import__("traceback").format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
