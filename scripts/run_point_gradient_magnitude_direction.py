"""Single-seed privileged decomposition of terminal Q-gradient errors.

Four PointObstacle QAM branches share the audited long-MC-v2 initialization,
offline transitions, TD/AM random streams and evaluation banks. Only the
terminal action-Q-gradient label differs. The MC reference is recomputed under
each privileged branch's current Actor and is never used for critic training.
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
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "point_gradient_magnitude_direction.json"
sys.path.insert(0, str(ROOT / "scripts"))
import run_point_long_training as longrun

M = longrun.M
CORE = M.core
BRANCHES = ("learned", "magnitude_only", "direction_only", "full_reference")
torch.set_num_threads(4)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def load_effective_config(protocol):
    c, _ = M.load_config()
    c = replace(c,
                actor_updates=protocol["stage1_actor_updates"],
                curve_updates=tuple(protocol["stage1_curve_updates"]),
                curve_episodes=protocol["curve_episodes"],
                final_episodes=protocol["final_episodes"])
    return c


def validate_source(c, protocol):
    source = ROOT / protocol["source_run"]
    seed_dir = source / f"seed-{protocol['seed']}"
    needed = ("protocol.json", "complete.json")
    for name in needed:
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    source_doc = json.loads((source / "protocol.json").read_text())
    previous = source_doc["config"]
    manifest_path = seed_dir / "initialization-source.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        input_dir = Path(manifest["source"])
    else:
        manifest = {"source": str(seed_dir), "hashes": {}}
        input_dir = seed_dir
    required_inputs = (protocol["initialization_source"].split("/", 1)[1],
                       protocol["train_data_source"].split("/", 1)[1],
                       protocol["valid_data_source"].split("/", 1)[1])
    for name in required_inputs:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        expected_hash = manifest.get("hashes", {}).get(name)
        if expected_hash and sha256(path) != expected_hash:
            raise RuntimeError(f"initialization-source manifest hash mismatch: {path}")
    # Reuse exactly the code and environment-critical sources audited by long-MC-v2.
    for rel in ("scripts/run_point_long_training.py",
                "scripts/point_reference_accelerated.py",
                "notebooks/02_qam_point_obstacle.py",
                "notebooks/02_qam_td_terminal_error.py",
                "configs/qam_point_obstacle.json"):
        expected_hash = source_doc.get("source_hashes", {}).get(rel)
        if expected_hash and sha256(ROOT / rel) != expected_hash:
            raise RuntimeError(f"long-MC-v2 source file changed: {rel}")
    allowed_overrides = {"actor_updates", "curve_updates", "curve_episodes", "final_episodes"}
    mismatches = {}
    for key, value in asdict(c).items():
        if key.startswith("reference_") or key in allowed_overrides:
            continue
        prior = previous[key]
        current = json.loads(json.dumps(value))
        if current != prior:
            mismatches[key] = {"source": prior, "experiment": current}
    if mismatches:
        raise RuntimeError(f"Task/architecture config differs from frozen source: {mismatches}")
    complete = json.loads((seed_dir / "complete.json").read_text())
    if int(complete["seed"]) != int(protocol["seed"]):
        raise RuntimeError("source seed metadata mismatch")
    return source, seed_dir, input_dir, previous, manifest


def new_branch(base, initial_critic, initial_target, c, seed):
    actor = copy.deepcopy(base)
    for p in actor.parameters():
        p.requires_grad_(True)
    critic = copy.deepcopy(initial_critic)
    target = copy.deepcopy(initial_target)
    return dict(actor=actor, critic=critic, target=target,
                aopt=torch.optim.Adam(actor.parameters(), lr=c.actor_lr),
                copt=torch.optim.Adam(critic.parameters(), lr=c.critic_lr),
                rng=CORE.generator(M.seed_for(seed, 701)))


def state_payload(states, completed, config_hash):
    payload = {"completed_updates": int(completed), "config_hash": config_hash,
               "branches": {}}
    for name, state in states.items():
        payload["branches"][name] = {
            key: state[key].state_dict() for key in ("actor", "critic", "target")
        }
        payload["branches"][name]["actor_optimizer"] = state["aopt"].state_dict()
        payload["branches"][name]["critic_optimizer"] = state["copt"].state_dict()
        payload["branches"][name]["rng"] = state["rng"].get_state()
    return payload


def save_suite_state(path, states, completed, config_hash):
    tmp = path.with_suffix(".tmp")
    torch.save(state_payload(states, completed, config_hash), tmp)
    tmp.replace(path)


def restore_suite_state(path, states, config_hash):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["config_hash"] != config_hash:
        raise RuntimeError("resume protocol/source hash mismatch")
    if set(saved["branches"]) != set(BRANCHES):
        raise RuntimeError("resume branch set mismatch")
    for name, state in states.items():
        branch = saved["branches"][name]
        for key in ("actor", "critic", "target"):
            state[key].load_state_dict(branch[key])
        state["aopt"].load_state_dict(branch["actor_optimizer"])
        state["copt"].load_state_dict(branch["critic_optimizer"])
        state["rng"].set_state(branch["rng"])
    assert_common_rng(states)
    return int(saved["completed_updates"])


def assert_common_rng(states):
    first = states[BRANCHES[0]]["rng"].get_state()
    for name in BRANCHES[1:]:
        if not torch.equal(first, states[name]["rng"].get_state()):
            raise RuntimeError(f"training RNG streams diverged at branch {name}")


def critic_action_gradient(critic, k, y, endpoint):
    with torch.enable_grad():
        a = endpoint.detach().requires_grad_(True)
        q = critic(k, y, a).mean(-1)
        grad = torch.autograd.grad(q.sum(), a)[0]
    if not bool(torch.isfinite(grad).all()):
        raise FloatingPointError("non-finite learned Q action gradient")
    return grad.detach()


def decompose_labels(g_learned, g_reference, zero_tolerance=1e-8):
    """Four prescribed terminal gradients, with zero vectors left at zero."""
    if g_learned.shape != g_reference.shape or g_learned.ndim != 2:
        raise ValueError("gradients must have matching [batch, action_dim] shape")
    if not bool(torch.isfinite(g_learned).all()) or not bool(torch.isfinite(g_reference).all()):
        raise FloatingPointError("non-finite gradient label")
    n_l = g_learned.norm(dim=-1, keepdim=True)
    n_r = g_reference.norm(dim=-1, keepdim=True)
    l_nonzero = n_l > zero_tolerance
    r_nonzero = n_r > zero_tolerance
    g_mag = torch.where(l_nonzero, n_r * g_learned / n_l.clamp_min(zero_tolerance),
                        torch.zeros_like(g_learned))
    g_dir = torch.where(r_nonzero, n_l * g_reference / n_r.clamp_min(zero_tolerance),
                        torch.zeros_like(g_reference))
    return {"learned": g_learned.detach(),
            "magnitude_only": g_mag.detach(),
            "direction_only": g_dir.detach(),
            "full_reference": g_reference.detach()}


def cosine_rows(a, b, eps=1e-12):
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    c = (a * b).sum(-1) / (na * nb).clamp_min(eps)
    return torch.where((na > eps) & (nb > eps), c.clamp(-1., 1.), torch.zeros_like(c))


def vector_metrics(g_learned, g_reference, labels, eps):
    nl, nr = g_learned.norm(dim=-1), g_reference.norm(dim=-1)
    cosine = cosine_rows(g_learned, g_reference)
    rows = []
    for i in range(len(g_learned)):
        row = dict(sample_index=int(i), learned_gradient_norm=float(nl[i]),
                   reference_gradient_norm=float(nr[i]),
                   learned_reference_cosine=float(cosine[i]),
                   sample_learned_to_reference_norm_ratio=float(nl[i] / (nr[i] + eps)),
                   learned_zero=bool(nl[i] <= eps), reference_zero=bool(nr[i] <= eps))
        for name in ("learned", "magnitude_only", "direction_only", "full_reference"):
            row[f"sample_relative_error_{name}"] = float(
                (labels[name][i] - g_reference[i]).norm() / (nr[i] + eps))
            row[f"gradient_norm_{name}"] = float(labels[name][i].norm())
        rows.append(row)
    summary = dict(
        count=len(rows),
        learned_gradient_norm_mean=float(nl.mean()),
        reference_gradient_norm_mean=float(nr.mean()),
        learned_reference_cosine_mean=float(cosine.mean()),
        learned_reference_cosine_median=float(cosine.median()),
        fraction_cos_lt_0=float((cosine < 0).float().mean()),
        fraction_cos_lt_0_5=float((cosine < .5).float().mean()),
        # A mean of rowwise ratios explodes when individual reference gradients
        # are near zero. Keep those raw ratios in the per-sample CSV, but make
        # the batch summary the ratio of mean norms.
        learned_to_reference_norm_ratio=float(nl.mean() / (nr.mean() + eps)),
        learned_zero_fraction=float((nl <= eps).float().mean()),
        reference_zero_fraction=float((nr <= eps).float().mean()))
    for name in ("learned", "magnitude_only", "direction_only", "full_reference"):
        errors = np.asarray([row[f"sample_relative_error_{name}"] for row in rows])
        summary[f"relative_error_{name}"] = float(errors.mean())
        summary[f"relative_error_{name}_median"] = float(np.median(errors))
        summary[f"relative_error_{name}_p90"] = float(np.quantile(errors, .90))
        summary[f"{name}_zero_fraction"] = float(np.mean([
            row[f"gradient_norm_{name}"] <= eps for row in rows]))
    for row in rows:
        row.update({key: value for key, value in summary.items() if key != "count"})
    return rows, summary


def restore_sample_error_metrics(rows, eps, zero_tolerance):
    """Recover rowwise errors from saved 2-D norms/cosines if older CSVs overwrote them.

    Earlier experiment output copied batch summaries onto each sample row using
    the same keys as the rowwise errors. The saved gradient norms and cosine are
    sufficient to reconstruct these distances without rerunning MC.
    """
    groups = {}
    for row in rows:
        nl = float(row["learned_gradient_norm"])
        nr = float(row["reference_gradient_norm"])
        cosine = float(row["learned_reference_cosine"])
        key = (row.get("method"), row.get("completed_updates"))
        groups.setdefault(key, []).append(row)
        d2 = max(0.0, nl * nl + nr * nr - 2.0 * nl * nr * cosine)
        row["sample_learned_to_reference_norm_ratio"] = nl / (nr + eps)
        row["sample_relative_error_learned"] = d2 ** .5 / (nr + eps)
        if nl <= zero_tolerance:
            error_mag = nr
        else:
            error_mag = max(0.0, 2.0 * nr * nr * (1.0 - cosine)) ** .5
        row["sample_relative_error_magnitude_only"] = error_mag / (nr + eps)
        if nr <= zero_tolerance:
            error_dir = nl
        else:
            error_dir = abs(nl - nr)
        row["sample_relative_error_direction_only"] = error_dir / (nr + eps)
        row["sample_relative_error_full_reference"] = 0.0
    # Rewrite the legacy duplicated summary ratio with its clarified meaning.
    for group in groups.values():
        mean_l = np.mean([float(row["learned_gradient_norm"]) for row in group])
        mean_r = np.mean([float(row["reference_gradient_norm"]) for row in group])
        batch_ratio = float(mean_l / (mean_r + eps))
        for row in group:
            row["learned_to_reference_norm_ratio"] = batch_ratio
    return rows


def mc_reference_batch(task, actor, base, c, seed, update, k, y, endpoint,
                       episodes, device, role=600000):
    """Call the unchanged audited longrun.independent_mc estimator per endpoint."""
    policy = copy.deepcopy(actor).to(device).eval()
    grads, ses, values, value_ses = [], [], [], []
    started = time.monotonic()
    for i in range(len(k)):
        mc_seed = M.seed_for(seed, role + int(update) * 1000 + i)
        result = longrun.independent_mc(
            task, policy, base, int(k[i]), y[i].detach().cpu().numpy(),
            endpoint[i].detach().cpu().numpy(), episodes, mc_seed, device)
        grads.append(result["gradient"])
        ses.append(result["gradient_se"])
        values.append(result["q"])
        value_ses.append(result["q_se"])
    return (torch.as_tensor(np.asarray(grads), dtype=torch.float32),
            np.asarray(ses), np.asarray(values), np.asarray(value_ses),
            time.monotonic() - started)


def diagnostic_batch(train, c, seed):
    rng = CORE.generator(M.seed_for(seed, 790))
    ids = torch.randint(len(train["k"]), (c.actor_batch,), generator=rng)
    k, y = train["k"][ids], train["y"][ids]
    z = torch.randn((len(ids), c.dimension), generator=rng)
    flow = torch.randn((len(ids), c.flow_steps, c.dimension), generator=rng)
    return dict(k=k, y=y, z=z, flow=flow, ids=ids)


def save_gradient_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def zero_step_sanity(task, states, base, c, train, seed, protocol, output, device, diag):
    """Verify all four labels at N=0 and write zero-vector fractions explicitly."""
    assert_common_rng(states)
    k, y, z, flow = diag["k"], diag["y"], diag["z"], diag["flow"]
    first = states["learned"]
    paths, _ = CORE.rollout(first["actor"], base, c, k, y, z, flow)
    endpoint = paths[:, -1]
    g_l = critic_action_gradient(first["critic"], k, y, endpoint)
    g_r, g_se, _, _, seconds = mc_reference_batch(
        task, first["actor"], base, c, seed, 0, k, y, endpoint,
        protocol["reference_mc_episodes"], device, role=610000)
    labels = decompose_labels(g_l, g_r, protocol["gradient_zero_tolerance"])
    tol = protocol["gradient_zero_tolerance"]
    n_l, n_r = g_l.norm(dim=-1), g_r.norm(dim=-1)
    both = (n_l > tol) & (n_r > tol)
    if bool(both.any()):
        assert torch.allclose(labels["magnitude_only"].norm(dim=-1)[both],
                              n_r[both], atol=2e-6, rtol=2e-6)
        assert bool((cosine_rows(labels["magnitude_only"], g_l)[both] > 1-2e-6).all())
        assert torch.allclose(labels["direction_only"].norm(dim=-1)[both],
                              n_l[both], atol=2e-6, rtol=2e-6)
        assert bool((cosine_rows(labels["direction_only"], g_r)[both] > 1-2e-6).all())
    assert torch.equal(labels["learned"], g_l)
    assert torch.equal(labels["full_reference"], g_r)
    samples, summary = vector_metrics(g_l, g_r, labels,
                                      protocol["gradient_relative_error_epsilon"])
    for row, se in zip(samples, g_se):
        row["reference_gradient_se_x"] = float(se[0])
        row["reference_gradient_se_y"] = float(se[1])
    write_json(output / "zero-step-gradient-sanity.json", dict(
        passed=True, seed=seed, endpoints=len(k), mc_episodes_per_endpoint=protocol["reference_mc_episodes"],
        mc_seconds=seconds, summary=summary,
        zero_vector_fraction={key: float(np.mean([r[f"{key}_zero"] for r in samples]))
                              for key in ("learned", "reference")}))
    # Explicit per-label fractions (the summary above also contains them).
    result = json.loads((output / "zero-step-gradient-sanity.json").read_text())
    result["zero_vector_fraction"] = {
        "learned_gradient": float((n_l <= tol).float().mean()),
        "reference_gradient": float((n_r <= tol).float().mean()),
        **{f"{name}_label": float((labels[name].norm(dim=-1) <= tol).float().mean())
           for name in BRANCHES}}
    result.pop("labels_zero_fraction", None)
    write_json(output / "zero-step-gradient-sanity.json", result)
    save_gradient_rows(output / "zero-step-gradient-samples.csv", samples)
    return dict(k=k, y=y, z=z, flow=flow, ids=diag["ids"]), summary


def evaluate_curves(task, states, base, curve_bank, c, N, output, curves,
                    source_seed_dir):
    replay_rows = load_rows(output / "learned-replay-validation.csv")
    for method in BRANCHES:
        if any(row["method"] == method and int(row["N"]) == N for row in curves):
            continue
        stats = M.evaluate(task, states[method]["actor"], base, curve_bank)
        if method == "learned":
            prior = pd.read_csv(source_seed_dir / "curves.csv")
            prior = prior[(prior.method == "learned") & (prior.N.astype(int) == int(N))]
            if len(prior) > 1:
                raise RuntimeError(f"source learned curve duplicated at N={N}")
            if len(prior) == 1:
                differences = {}
                for key in ("success_rate", "collision_rate", "timeout_rate", "discounted_return"):
                    difference = float(stats[key]) - float(prior.iloc[0][key])
                    differences[key] = difference
                    if not np.isclose(difference, 0., atol=1e-12, rtol=0.):
                        raise RuntimeError(
                            f"Learned reproduction mismatch at N={N} for {key}: "
                            f"source={prior.iloc[0][key]} current={stats[key]}")
                replay_rows = [r for r in replay_rows if int(r["N"]) != N]
                replay_rows.append(dict(N=N, source_curve_available=True, passed=True,
                                        max_abs_metric_difference=max(map(abs, differences.values()))))
            else:
                replay_rows = [r for r in replay_rows if int(r["N"]) != N]
                replay_rows.append(dict(N=N, source_curve_available=False, passed=None,
                                        max_abs_metric_difference=None))
            pd.DataFrame(replay_rows).sort_values("N").to_csv(
                output / "learned-replay-validation.csv", index=False)
        curves.append(dict(seed=59001, method=method, N=N,
                           actor_updates=N, td_updates=N, **stats))
        save_branch_checkpoint(output / "checkpoints" / f"{method}-N{N}.pt",
                               states[method], N, method)
    curves.sort(key=lambda row: (int(row["N"]), BRANCHES.index(row["method"])))
    pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)


def save_branch_checkpoint(path, state, N, method):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(N=int(N), method=method,
                    actor=state["actor"].state_dict(),
                    critic=state["critic"].state_dict(),
                    target=state["target"].state_dict(),
                    actor_optimizer=state["aopt"].state_dict(),
                    critic_optimizer=state["copt"].state_dict(),
                    rng=state["rng"].get_state()), path)


def virtual_actor_update_diagnostic(task, states, base, c, train, seed, N,
                                    protocol, diag, output, device):
    """Compare q and Adam deltas for four labels on one Learned-branch state."""
    state = states["learned"]
    k, y, z, flow = diag["k"], diag["y"], diag["z"], diag["flow"]
    paths, _ = CORE.rollout(state["actor"], base, c, k, y, z, flow)
    endpoint = paths[:, -1]
    g_l = critic_action_gradient(state["critic"], k, y, endpoint)
    g_r, g_se, _, _, seconds = mc_reference_batch(
        task, state["actor"], base, c, seed, N, k, y, endpoint,
        protocol["reference_mc_episodes"], device, role=620000)
    labels = decompose_labels(g_l, g_r, protocol["gradient_zero_tolerance"])
    qs, losses, steps = {}, {}, {}
    for method in BRANCHES:
        terminal = -c.beta * labels[method]
        adj = CORE.lean_adjoint(base, c, k, y, paths, terminal)
        q, loss = CORE.actor_gradient(state["actor"], base, c, k, y, paths, adj)
        qs[method], losses[method] = q.detach().clone(), loss
        steps[method] = CORE.virtual_adam(state["actor"], state["aopt"], q, c).detach().clone()
    q_ref, step_ref = qs["full_reference"], steps["full_reference"]
    rows = []
    for method in BRANCHES:
        rows.append(dict(
            N=N, method=method, samples=len(k), mc_seconds=seconds,
            matching_loss=losses[method], parameter_gradient_norm=float(qs[method].norm()),
            q_cosine_to_full_reference=float(cosine_rows(qs[method][None], q_ref[None])[0]),
            q_relative_distance_to_full_reference=float(
                (qs[method] - q_ref).norm() / (q_ref.norm() + 1e-12)),
            virtual_adam_step_norm=float(steps[method].norm()),
            adam_step_cosine_to_full_reference=float(
                cosine_rows(steps[method][None], step_ref[None])[0]),
            adam_step_relative_distance_to_full_reference=float(
                (steps[method] - step_ref).norm() / (step_ref.norm() + 1e-12)),
            step_zero=bool(steps[method].norm() <= 1e-12)))
    all_rows = []
    path = output / "actor_update_decomposition.csv"
    if path.exists():
        all_rows.extend(pd.read_csv(path).to_dict("records"))
    all_rows = [row for row in all_rows if int(row["N"]) != N]
    all_rows.extend(rows)
    pd.DataFrame(all_rows).sort_values(["N", "method"]).to_csv(path, index=False)
    write_json(output / "actor-update-decomposition" / f"N{N}.json", dict(
        N=N, seed=seed, source_actor_branch="learned", current_policy_reference=True,
        reference_mc_episodes=protocol["reference_mc_episodes"], mc_seconds=seconds,
        gradient_norms={name: float(labels[name].norm(dim=-1).mean()) for name in BRANCHES},
        raw_reference_gradient_se_rms=float(np.sqrt(np.mean(g_se ** 2))), rows=rows))
    return rows


def train_update(task, data, base, state, c, seed, update, method, protocol,
                 device, training_logs, gradient_samples):
    actor, critic, target, rng = (state["actor"], state["critic"],
                                  state["target"], state["rng"])
    # Critic always gets its ordinary TD update before the Actor label is formed.
    critic_ids = torch.randint(len(data["k"]), (c.critic_batch,), generator=rng)
    td_loss, q_mean = M.td_step(critic, target, state["copt"], data,
                                critic_ids, rng, c, actor, base)
    actor_ids = torch.randint(len(data["k"]), (c.actor_batch,), generator=rng)
    k, y = data["k"][actor_ids], data["y"][actor_ids]
    z = torch.randn((len(actor_ids), c.dimension), generator=rng)
    flow = torch.randn((len(actor_ids), c.flow_steps, c.dimension), generator=rng)
    paths, _ = CORE.rollout(actor, base, c, k, y, z, flow)
    endpoint = paths[:, -1]
    g_l = critic_action_gradient(critic, k, y, endpoint)
    log_this_update = (update + 1) % protocol["gradient_log_interval"] == 0
    # Learned branch's MC is strictly a read-only metric, never its Actor label.
    use_reference = method != "learned" or log_this_update
    if use_reference:
        g_r, g_se, _, _, mc_seconds = mc_reference_batch(
            task, actor, base, c, seed, update, k, y, endpoint,
            protocol["reference_mc_episodes"], device, role=630000)
        labels = decompose_labels(g_l, g_r, protocol["gradient_zero_tolerance"])
        gradient = labels[method]
    else:
        g_r = g_se = None
        mc_seconds = 0.
        gradient = g_l

    terminal = -c.beta * gradient
    adjoints = CORE.lean_adjoint(base, c, k, y, paths, terminal)
    q, loss = CORE.actor_gradient(actor, base, c, k, y, paths, adjoints)
    CORE.set_gradient(actor, q)
    clipped_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), c.actor_clip,
                                                 error_if_nonfinite=True)
    state["aopt"].step()
    actor.zero_grad(set_to_none=True)

    row = dict(seed=seed, method=method, completed_updates=update + 1,
               td_updates=update + 1, actor_updates=update + 1,
               td_loss=td_loss, td_q_mean=q_mean, matching_loss=loss,
               actor_parameter_gradient_norm=float(q.norm()),
               actor_gradient_preclip_norm=float(clipped_norm),
               reference_mc_seconds=mc_seconds)
    if log_this_update:
        rows, summary = vector_metrics(g_l, g_r, labels,
                                      protocol["gradient_relative_error_epsilon"])
        for item in rows:
            item.update(seed=seed, method=method, completed_updates=update + 1,
                        reference_gradient_se_x=float(g_se[item["sample_index"], 0]),
                        reference_gradient_se_y=float(g_se[item["sample_index"], 1]))
        gradient_samples.extend(rows)
        row.update(summary)
        row["reference_gradient_se_rms"] = float(np.sqrt(np.mean(g_se ** 2)))
    training_logs.append(row)
    return row


def evaluate_table(task, states, base, bank, N):
    rows = []
    for method in BRANCHES:
        stats = M.evaluate(task, states[method]["actor"], base, bank)
        rows.append(dict(seed=59001, N=N, method=method, **stats))
    return rows


def save_test50(task, states, base, c, protocol, N, parent, folder_name=None):
    folder = parent / (folder_name or f"test50-N{N}")
    folder.mkdir(parents=True, exist_ok=True)
    bank = M.noise_bank(c, protocol["paired_episodes"], M.seed_for(protocol["seed"], 505))
    outcomes = {}
    all_paths = {}
    rows = []
    for method in BRANCHES:
        stats, paths, status = M.evaluate(task, states[method]["actor"], base, bank,
                                          trajectories=True)
        outcomes[method] = status
        all_paths[method] = paths
        rows.append(dict(seed=protocol["seed"], N=N, method=method, **stats))
        plot_trajectories(paths, status, task.c, folder / f"{method}.png",
                          f"{method} · N={N} · 50 paired episodes")
    paired = pd.DataFrame({"episode": np.arange(protocol["paired_episodes"]),
                           **{method: outcomes[method] for method in BRANCHES}})
    for method in BRANCHES:
        paired[f"{method}_success"] = paired[method] == "success"
        paired[f"{method}_collision"] = paired[method] == "collision"
    paired.to_csv(folder / "paired_outcomes.csv", index=False)
    np.savez_compressed(folder / "trajectories.npz",
                        **{f"paths_{m}": np.transpose(all_paths[m], (1, 0, 2)) for m in BRANCHES},
                        **{f"status_{m}": outcomes[m] for m in BRANCHES},
                        starts=bank["start"], environment_noise=bank["env"],
                        z=bank["z"], flow=bank["flow"])
    plot_overlay(all_paths, outcomes, task.c, folder / "all_four_overlay.png",
                 f"All four branches · N={N} · same 50 exogenous episodes")
    pd.DataFrame(rows).to_csv(folder / "branch-summary.csv", index=False)
    counts = paired_transition_counts(paired)
    pd.DataFrame(counts).to_csv(folder / "paired-transition-counts.csv", index=False)
    return rows, counts, folder


def paired_transition_counts(paired):
    rows = []
    learned = paired["learned"].to_numpy()
    for name in ("direction_only", "magnitude_only", "full_reference"):
        other = paired[name].to_numpy()
        rows.append(dict(comparison=f"learned_fail_to_{name}_success",
                         count=int(np.sum((learned != "success") & (other == "success")))))
        rows.append(dict(comparison=f"learned_success_to_{name}_fail",
                         count=int(np.sum((learned == "success") & (other != "success")))))
    direction = paired["direction_only"].to_numpy()
    rows.append(dict(comparison="learned_collision_to_direction_success",
                     count=int(np.sum((learned == "collision") & (direction == "success")))))
    for name in ("direction_only", "magnitude_only", "full_reference"):
        other = paired[name].to_numpy()
        rows.append(dict(comparison=f"learned_collision_to_{name}_success",
                         count=int(np.sum((learned == "collision") & (other == "success")))))
    return rows


def plot_trajectories(paths, status, c, path, title):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    add_task_geometry(ax, c)
    for i in range(paths.shape[1]):
        end = np.flatnonzero(np.isfinite(paths[:, i, 0]))
        if not len(end):
            continue
        xy = paths[:end[-1] + 1, i]
        color = "#2b8cbe" if status[i] == "success" else (
            "#d7301f" if status[i] == "collision" else "#756bb1")
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=.58, lw=.8)
        ax.scatter(xy[0, 0], xy[0, 1], marker="s", c="#222222", s=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_overlay(paths, status, c, path, title):
    colors = {"learned": "#d95f02", "magnitude_only": "#7570b3",
              "direction_only": "#1b9e77", "full_reference": "#377eb8"}
    fig, ax = plt.subplots(figsize=(7, 6))
    add_task_geometry(ax, c)
    for method in BRANCHES:
        for i in range(paths[method].shape[1]):
            end = np.flatnonzero(np.isfinite(paths[method][:, i, 0]))
            if len(end):
                xy = paths[method][:end[-1] + 1, i]
                ax.plot(xy[:, 0], xy[:, 1], color=colors[method], alpha=.26,
                        lw=.8, label=method if i == 0 else None)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def add_task_geometry(ax, c):
    ax.add_patch(Circle(c.obstacle_center, c.obstacle_radius,
                        color="#666666", alpha=.78, zorder=4, label="obstacle"))
    ax.add_patch(Circle(c.goal_center, c.goal_radius, facecolor="none",
                        edgecolor="#e6ab02", lw=2., zorder=5, label="goal"))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")
    ax.grid(alpha=.2)


def compute_mechanism_results(final_rows):
    df = pd.DataFrame(final_rows)
    output = []
    for N in sorted(df["N"].astype(int).unique()):
        part = df[df["N"].astype(int) == N].set_index("method")
        s_l = float(part.loc["learned", "success_rate"])
        s_r = float(part.loc["full_reference", "success_rate"])
        denominator = s_r - s_l
        for method in BRANCHES:
            row = part.loc[method]
            success = float(row["success_rate"])
            recovery = None if denominator <= 0 else float((success - s_l) / denominator)
            output.append(dict(
                N=N, method=method, success_percent=100 * success,
                collision_percent=100 * float(row["collision_rate"]),
                return_mean=float(row["discounted_return"]),
                delta_vs_learned_pp=100 * (success - s_l),
                gap_recovery=(0. if method == "learned" else
                              1. if method == "full_reference" and denominator > 0 else recovery),
                reference_gap_positive=denominator > 0, episodes=int(row["episodes"])))
    return output


def success_table(rows, N):
    part = {r["method"]: r for r in rows if int(r["N"]) == N}
    return {name: float(part[name]["success_rate"]) for name in BRANCHES}


def admission_decision(rows, protocol):
    values = success_table(rows, 100)
    s_l, s_m, s_d, s_r = (values[k] for k in BRANCHES)
    recovery = None if s_r <= s_l else (s_d - s_l) / (s_r - s_l)
    delta_pp = 100 * (s_d - s_l)
    passed = ((recovery is not None and
               recovery >= protocol["admission_direction_gap_recovery_min"])
              or delta_pp >= protocol["admission_direction_success_improvement_min_pp"])
    return dict(seed=59001, N=100, success_rate=values,
                direction_gap_recovery=recovery,
                direction_minus_learned_pp=delta_pp,
                magnitude_minus_learned_pp=100 * (s_m - s_l),
                full_reference_gap_positive=s_r > s_l,
                continue_to_300=bool(passed),
                rule=("continue iff direction recovery >= 0.50 or direction gain >= 3 pp; "
                      "otherwise stop at N=100"))


def relative_percent(value, digits=2):
    if value is None:
        return "undefined (S_R <= S_L)"
    return f"{100 * value:.{digits}f}%"


def report_text(final_rows, curves, admission, gradient_samples, actor_rows,
                test50_summaries, protocol, stage_complete):
    by_n = {}
    for row in final_rows:
        by_n.setdefault(int(row["N"]), {})[row["method"]] = row
    n_last = max(by_n)
    finals = by_n[n_last]
    n100 = by_n[100]
    sl100, sm100, sd100, sr100 = [float(n100[name]["success_rate"]) for name in BRANCHES]
    recovery100 = None if sr100 <= sl100 else (sd100 - sl100) / (sr100 - sl100)
    nlast_values = [float(finals[name]["success_rate"]) for name in BRANCHES]
    sl_last, sm_last, sd_last, sr_last = nlast_values
    recovery_last = None if sr_last <= sl_last else (sd_last - sl_last) / (sr_last - sl_last)
    actor_at_100 = {r["method"]: r for r in actor_rows if int(r["N"]) == 100}
    if actor_at_100:
        direction_update_closer = (
            actor_at_100["direction_only"]["adam_step_relative_distance_to_full_reference"] <
            actor_at_100["learned"]["adam_step_relative_distance_to_full_reference"] and
            actor_at_100["direction_only"]["adam_step_relative_distance_to_full_reference"] <
            actor_at_100["magnitude_only"]["adam_step_relative_distance_to_full_reference"])
        q_closer = (
            actor_at_100["direction_only"]["q_relative_distance_to_full_reference"] <
            actor_at_100["learned"]["q_relative_distance_to_full_reference"] and
            actor_at_100["direction_only"]["q_relative_distance_to_full_reference"] <
            actor_at_100["magnitude_only"]["q_relative_distance_to_full_reference"])
    else:
        direction_update_closer = q_closer = False
    actor_at_primary = {r["method"]: r for r in actor_rows if int(r["N"]) == n_last}
    if actor_at_primary:
        primary_step_closer = (
            actor_at_primary["direction_only"]["adam_step_relative_distance_to_full_reference"] <
            actor_at_primary["learned"]["adam_step_relative_distance_to_full_reference"] and
            actor_at_primary["direction_only"]["adam_step_relative_distance_to_full_reference"] <
            actor_at_primary["magnitude_only"]["adam_step_relative_distance_to_full_reference"])
        primary_q_closer = (
            actor_at_primary["direction_only"]["q_relative_distance_to_full_reference"] <
            actor_at_primary["learned"]["q_relative_distance_to_full_reference"] and
            actor_at_primary["direction_only"]["q_relative_distance_to_full_reference"] <
            actor_at_primary["magnitude_only"]["q_relative_distance_to_full_reference"])
    else:
        primary_step_closer = primary_q_closer = False
    primary_gain = sd_last - sl_last
    magnitude_gain = sm_last - sl_last
    dominant = (recovery_last is not None and recovery_last >= .5 and
                primary_gain > magnitude_gain and primary_step_closer and primary_q_closer)
    partial = (primary_gain > 0 and
               (primary_step_closer or primary_q_closer or primary_gain > magnitude_gain))
    supported = "supported" if dominant else "partially supported" if partial else "not supported"
    final_episodes = int(protocol["final_episodes"])
    direction_success_delta = int(round((sd100 - sl100) * final_episodes))
    direction_improvement_text = (
        f"有极小表观增加（{direction_success_delta}/{final_episodes} 个 episode），不足以视为实质改善"
        if sd100 > sl100 else "否，未高于 Learned")
    grad_at100 = [r for r in gradient_samples if int(r["completed_updates"]) == 100]
    grad_summary = {}
    for method in BRANCHES:
        points = [r for r in grad_at100 if r["method"] == method]
        if points:
            e_l = np.asarray([r["sample_relative_error_learned"] for r in points])
            e_m = np.asarray([r["sample_relative_error_magnitude_only"] for r in points])
            e_d = np.asarray([r["sample_relative_error_direction_only"] for r in points])
            nl = np.asarray([r["learned_gradient_norm"] for r in points])
            nr = np.asarray([r["reference_gradient_norm"] for r in points])
            cosine = np.asarray([r["learned_reference_cosine"] for r in points])
            grad_summary[method] = dict(
                learned_gradient_norm_mean=float(nl.mean()),
                reference_gradient_norm_mean=float(nr.mean()),
                learned_reference_cosine_mean=float(cosine.mean()),
                learned_to_reference_norm_ratio=float(nl.mean() / (nr.mean() +
                    float(protocol["gradient_relative_error_epsilon"]))),
                relative_error_learned=float(e_l.mean()),
                relative_error_magnitude_only=float(e_m.mean()),
                relative_error_direction_only=float(e_d.mean()),
                relative_error_learned_median=float(np.median(e_l)),
                relative_error_magnitude_only_median=float(np.median(e_m)),
                relative_error_direction_only_median=float(np.median(e_d)),
                relative_error_learned_p90=float(np.quantile(e_l, .90)),
                relative_error_magnitude_only_p90=float(np.quantile(e_m, .90)),
                relative_error_direction_only_p90=float(np.quantile(e_d, .90)),
                fraction_cos_lt_0=float(np.mean(cosine < 0)),
                fraction_cos_lt_0_5=float(np.mean(cosine < .5)))
    lines = [
        "# PointObstacle Q-gradient magnitude × direction mechanism experiment",
        "",
        "## 五个问题：先给结论",
        "",
        f"1. **Direction-only 是否提高闭环成功率？** N=100: {100*(sd100-sl100):+.2f} pp "
        f"({direction_improvement_text}); "
        f"final N={n_last}: {100*(sd_last-sl_last):+.2f} pp.",
        f"2. **Magnitude-only 是否提高闭环成功率？** N=100: {100*(sm100-sl100):+.2f} pp "
        f"({'是' if sm100 > sl100 else '否，未高于 Learned'}); "
        f"final N={n_last}: {100*(sm_last-sl_last):+.2f} pp.",
        f"3. **Direction-only 恢复 Full-reference gap 的比例？** N=100: "
        f"{relative_percent(recovery100)}; final N={n_last}: {relative_percent(recovery_last)}.",
        f"4. **经过 Q-gradient → terminal adjoint → Lean Adjoint → parameter gradient → Adam，"
        f"direction-only 是否更接近 full-reference update？** N=100 的 parameter-gradient "
        f"距离更接近: {'是' if q_closer else '否'}；虚拟 Adam-step 距离更接近: "
        f"{'是' if direction_update_closer else '否'}。",
        f"5. **当前证据是否支持 direction error 是主要可恢复失效来源？** **{supported}**。"
        f"仅单 seed；按 N=100 的预注册闭环差距与虚拟 Adam-step 结果判定，"
        f"不能推广到其他 seed/任务。",
        "",
        "## 设计与协议",
        "",
        f"单 seed `{protocol['seed']}`，源 run `{protocol['source_run']}`。四臂共享相同离线转移、"
        "Base、初始 Actor/Critic/target、Adam 初始化、TD/Actor 随机流、批次索引、Flow 噪声和评价噪声。"
        "Critic 在每臂继续普通 TD 更新；reference gradient 只用于 Actor terminal label，绝不进入 TD target、"
        "Critic loss、checkpoint 选择、奖励、Base 或评价。",
        "",
        "`magnitude_only` 保留 g_L 方向并替换为 ||g_R||；`direction_only` 保留 ||g_L|| 并替换为 g_R 方向；"
        "`full_reference` 使用 g_R；Learned 使用 g_L。MC reference 每 endpoint 为8192条 IID continuation，"
        "由 long-MC-v2 的同一个 `independent_mc` score estimator 估计，后续动作来自该 branch 当前 Actor。",
        "",
        f"第一阶段为 `{protocol['stage1_actor_updates']}` 次 Actor 和 TD 更新；检查点为 "
        f"{protocol['stage1_curve_updates']}，曲线每臂每点 {protocol['curve_episodes']} 回合，"
        f"N=100 final 每臂 {protocol['final_episodes']} 回合。",
        "口径说明：源 Long-MC-v2 中约 52.29% / 65.06% 是 N=300 的 final 对照；本机制实验按预先固定的准入规则停在 N=100，因此这里的四臂 final 数值是 N=100，不应当作同一训练步数直接对比。N=100 的 1024 回合 learned 曲线与源曲线精确重放；4096 回合 final 是本实验固定共同噪声 bank 上的独立评价。",
        "",
        "## N=100 admission 表",
        "",
        "| Method | Success % | Collision % | Return | Δ vs Learned | Gap recovery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in compute_mechanism_results([r for r in final_rows if int(r["N"]) == 100]):
        lines.append(f"| {row['method']} | {row['success_percent']:.2f} | "
                     f"{row['collision_percent']:.2f} | {row['return_mean']:.5f} | "
                     f"{row['delta_vs_learned_pp']:+.2f} pp | "
                     f"{relative_percent(row['gap_recovery'])} |")
    lines.extend(["", f"Admission: direction recovery `{relative_percent(admission['direction_gap_recovery'])}`; "
                  f"direction gain `{admission['direction_minus_learned_pp']:+.2f} pp`; "
                  f"predeclared continuation decision: **{'continue to N=300' if admission['continue_to_300'] else 'stop at N=100'}**.", ""])
    if n_last != 100:
        lines.extend(["## N=300 final comparison", "",
                      "| Method | Success % | Collision % | Return | Δ vs Learned | Gap recovery |",
                      "|---|---:|---:|---:|---:|---:|"])
        for row in compute_mechanism_results([r for r in final_rows if int(r["N"]) == n_last]):
            lines.append(f"| {row['method']} | {row['success_percent']:.2f} | "
                         f"{row['collision_percent']:.2f} | {row['return_mean']:.5f} | "
                         f"{row['delta_vs_learned_pp']:+.2f} pp | "
                         f"{relative_percent(row['gap_recovery'])} |")
        lines.append("")
    lines.extend(["## Gradient-label decomposition", "",
                  "下表为 N=100 前最后一批训练 label 的逐样本诊断汇总（每 branch 64 个点）。范数比为该批次 ||gL|| 均值除以 ||gR|| 均值；eL/eM/eD 的均值仍按逐样本相对误差直接平均，可能被 ||gR|| 接近零的样本放大，因此同时报告中位数与 p90，且保留原始 per-sample 项。"])
    if grad_summary:
        lines.extend(["", "| Branch | ||gL|| | ||gR|| | cosine mean | mean-norm ratio | mean eL/eM/eD | median eL/eM/eD | p90 eL/eM/eD | cos<0 | cos<0.5 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for method in BRANCHES:
            row = grad_summary.get(method)
            if row:
                lines.append(f"| {method} | {row['learned_gradient_norm_mean']:.4g} | "
                             f"{row['reference_gradient_norm_mean']:.4g} | "
                             f"{row['learned_reference_cosine_mean']:.3f} | "
                             f"{row['learned_to_reference_norm_ratio']:.3f} | "
                             f"{row['relative_error_learned']:.3g}/"
                             f"{row['relative_error_magnitude_only']:.3g}/"
                             f"{row['relative_error_direction_only']:.3g} | "
                             f"{row['relative_error_learned_median']:.3g}/"
                             f"{row['relative_error_magnitude_only_median']:.3g}/"
                             f"{row['relative_error_direction_only_median']:.3g} | "
                             f"{row['relative_error_learned_p90']:.3g}/"
                             f"{row['relative_error_magnitude_only_p90']:.3g}/"
                             f"{row['relative_error_direction_only_p90']:.3g} | "
                             f"{row['fraction_cos_lt_0']:.3f} | {row['fraction_cos_lt_0_5']:.3f} |")
    lines.extend(["", "原始 per-sample 向量范数、rowwise norm ratio、cosine、reference MC standard errors 与三种相对误差保存在 "
                  "`gradient-decomposition-per-sample.csv`；相对误差使用 protocol 中固定的 epsilon。旧格式结果中的逐样本误差列曾被批次均值覆盖，本报告按保存的 ||gL||、||gR|| 与 cosine 用欧氏距离恒等式恢复逐样本值，不重跑训练或 MC。zero-vector fraction 单独记录在 "
                  "`zero-step-gradient-sanity.json` 和逐样本 CSV 中。", "",
                  "## Lean Adjoint / Actor gradient / Adam 虚拟更新", "",
                  "所有虚拟比较都在同一个 Learned branch Actor、Critic、Base、paths 和 optimizer history 上完成，"
                  "不执行 optimizer.step。对每个终端标签构造 adjoint、matching parameter gradient，并用克隆的同一 Adam state 算虚拟一步。",
                  "", "| N | Method | cos(q,qR) | rel ||q−qR|| | cos(Adam step,ref) | rel Adam-step distance |",
                  "|---:|---|---:|---:|---:|---:|"])
    for row in actor_rows:
        lines.append(f"| {int(row['N'])} | {row['method']} | "
                     f"{row['q_cosine_to_full_reference']:.4f} | "
                     f"{row['q_relative_distance_to_full_reference']:.4f} | "
                     f"{row['adam_step_cosine_to_full_reference']:.4f} | "
                     f"{row['adam_step_relative_distance_to_full_reference']:.4f} |")
    lines.extend(["", "## 50 条配对轨迹", ""])
    for summary in test50_summaries:
        N = int(summary["N"])
        lines.append(f"### N={N}")
        lines.append("")
        lines.append("同一组初始状态、环境噪声、exogenous z/Flow 随机量。状态转移按任务定义以碰撞/成功/超时终止。")
        lines.append("")
        lines.append("| Method | Success | Collision | Return |")
        lines.append("|---|---:|---:|---:|")
        for method in BRANCHES:
            r = next(row for row in summary["summary"] if row["method"] == method)
            lines.append(f"| {method} | {100*r['success_rate']:.1f}% | "
                         f"{100*r['collision_rate']:.1f}% | {r['discounted_return']:.4f} |")
        lines.append("")
        lines.append("Paired transition counts: `" + json.dumps(summary["counts"], ensure_ascii=False) + "`.")
        lines.append("")
    lines.extend(["## 预先固定的解释边界", "",
                  "- `direction_only ≈ full_reference` 且 `magnitude_only ≈ learned` 才支持本 seed/task 的方向主导解释。",
                  "- `direction_only > learned` 但远低于 full reference 表示方向有帮助但不足。",
                  "- `direction_only ≈ learned < full_reference` 表示离线方向误差改善没有转化为闭环因果收益。",
                  "- 若 magnitude-only 优于 direction-only，记录幅值侧更受支持；不事后改写。",
                  "- 本实验只有一个探索 seed；没有新算法、无调参、无 best-checkpoint 选择。结论不能推广。", "",
                  "## 文件与状态", "",
                  "- `mechanism_results.csv`: 主结果表。",
                  "- `curves.csv`: 共同1024 episode噪声下的各固定 checkpoint 曲线。",
                  "- `learned-replay-validation.csv`: 对源 run 已有检查点的 Learned 臂逐指标精确重放；源未记录 N=75 时明确标记为 unavailable。",
                  "- `gradient-decomposition-per-sample.csv`: 每10次更新的原始梯度诊断。",
                  "- `actor_update_decomposition.csv`: N=0/50/100及（若准入）N=150/300的虚拟梯度与Adam分析。",
                  "- `test50-N*/`: 四臂图片、overlay、配对 outcome CSV 和轨迹 NPZ。",
                  f"- Stage status: `{stage_complete}`。", ""])
    return "\n".join(lines)


def write_report(output, final_rows, curves, admission, gradient_samples,
                 actor_rows, test50_summaries, protocol, stage_complete):
    gradient_samples = restore_sample_error_metrics(
        gradient_samples, protocol["gradient_relative_error_epsilon"],
        protocol["gradient_zero_tolerance"])
    pd.DataFrame(gradient_samples).to_csv(
        output / "gradient-decomposition-per-sample.csv", index=False)
    mechanism = compute_mechanism_results(final_rows)
    pd.DataFrame(mechanism).to_csv(output / "mechanism_results.csv", index=False)
    (output / "report.md").write_text(report_text(final_rows, curves, admission,
                                                    gradient_samples, actor_rows,
                                                    test50_summaries, protocol,
                                                    stage_complete))


def initialize_run(protocol, output, device):
    c = load_effective_config(protocol)
    source, seed_dir, input_dir, source_config, input_manifest = validate_source(c, protocol)
    inputs_dir = output / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    input_names = ("shared-initialization.pt", "train-transitions.npz", "valid-transitions.npz")
    for name in input_names:
        src = input_dir / name
        dst = inputs_dir / name
        if dst.exists():
            if sha256(dst) != sha256(src):
                raise RuntimeError(f"existing experiment input differs from frozen source: {dst}")
        else:
            shutil.copy2(src, dst)
    init_path = inputs_dir / "shared-initialization.pt"
    train_path = inputs_dir / "train-transitions.npz"
    valid_path = inputs_dir / "valid-transitions.npz"
    init = torch.load(init_path, map_location="cpu", weights_only=True)
    base = CORE.ConditionalVelocity(c)
    base.load_state_dict(init["base"])
    for p in base.parameters():
        p.requires_grad_(False)
    initial_critic = CORE.Critic(c)
    initial_critic.load_state_dict(init["critic"])
    initial_target = CORE.Critic(c)
    initial_target.load_state_dict(init["target"])
    train_np = np.load(train_path)
    train = M.as_torch({key: train_np[key] for key in train_np.files})
    valid_np = np.load(valid_path)
    valid = M.as_torch({key: valid_np[key] for key in valid_np.files})
    states = {name: new_branch(base, initial_critic, initial_target, c, protocol["seed"])
              for name in BRANCHES}
    hashes = {name: M.model_hash(states["learned"][name])
              for name in ("actor", "critic", "target")}
    for branch in BRANCHES[1:]:
        for name in ("actor", "critic", "target"):
            if M.model_hash(states[branch][name]) != hashes[name]:
                raise RuntimeError(f"non-identical initialization: {branch}/{name}")
    assert_common_rng(states)
    # Adam states are fresh and must be identical across branches.
    for branch in BRANCHES[1:]:
        if states[branch]["aopt"].state_dict() != states["learned"]["aopt"].state_dict():
            raise RuntimeError("actor optimizer initialization differs")
        if states[branch]["copt"].state_dict() != states["learned"]["copt"].state_dict():
            raise RuntimeError("critic optimizer initialization differs")
    source_hashes = {"protocol.json": sha256(source / "protocol.json"),
                     "complete.json": sha256(seed_dir / "complete.json"),
                     "initialization-source.json": sha256(seed_dir / "initialization-source.json"),
                     "initialization-input-directory": str(input_dir),
                     "shared-initialization.pt": sha256(init_path),
                     "train-transitions.npz": sha256(train_path),
                     "valid-transitions.npz": sha256(valid_path),
                     "run_point_long_training.py": sha256(ROOT / "scripts/run_point_long_training.py"),
                     "point_reference_accelerated.py": sha256(ROOT / "scripts/point_reference_accelerated.py"),
                     "point_obstacle_notebook_source.py": sha256(ROOT / "notebooks/02_qam_point_obstacle.py"),
                     "qam_core.py": sha256(ROOT / "notebooks/02_qam_td_terminal_error.py"),
                     "experiment_runner.py": sha256(Path(__file__)),
                     "experiment_config.json": sha256(CONFIG_PATH),
                     "effective-config": hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode()).hexdigest()}
    payload = dict(protocol=protocol, config=asdict(c), source_hashes=source_hashes,
                   source_run=protocol["source_run"], source_branch_initial_hashes=hashes,
                   python=sys.version,
                   torch=torch.__version__, numpy=np.__version__, device=str(device),
                   gpu=(torch.cuda.get_device_name() if str(device).startswith("cuda") else None))
    config_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if old["config_hash"] != config_hash:
            raise RuntimeError("output directory exists with a different frozen protocol")
    else:
        write_json(output / "protocol.json", dict(**payload, config_hash=config_hash,
                   created_at=time.time()))
    return c, source, seed_dir, train, valid, base, initial_critic, initial_target, states, config_hash


def load_rows(path):
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def run_stage(task, data, base, states, c, seed, protocol, output, config_hash,
              source_seed_dir, target_updates, diag, device):
    checkpoint = output / "suite-latest.pt"
    completed = restore_suite_state(checkpoint, states, config_hash) if checkpoint.exists() else 0
    if completed > target_updates:
        return completed
    curves = load_rows(output / "curves.csv")
    train_logs = load_rows(output / "training-log.csv")
    grad_samples = load_rows(output / "gradient-decomposition-per-sample.csv")
    actor_rows = load_rows(output / "actor_update_decomposition.csv")
    curve_bank = M.noise_bank(c, protocol["curve_episodes"], M.seed_for(seed, 501))
    curve_updates = set(protocol["stage1_curve_updates"] + protocol["stage2_curve_updates"])
    diag_updates = set(protocol["actor_update_diagnostic_updates_stage1"] +
                        protocol["actor_update_diagnostic_updates_stage2"])
    if completed == 0:
        # A fixed diagnostic batch is reused at every virtual-update checkpoint.
        diag_state, _ = zero_step_sanity(task, states, base, c, data, seed,
                                         protocol, output, device, diag)
        write_json(output / "diagnostic-batch.json", dict(
            indices=diag_state["ids"].tolist(), k=diag_state["k"].tolist(),
            y=diag_state["y"].tolist(), z=diag_state["z"].tolist(),
            flow=diag_state["flow"].tolist()))
    else:
        saved_diag = json.loads((output / "diagnostic-batch.json").read_text())
        diag_state = dict(ids=torch.tensor(saved_diag["indices"], dtype=torch.long),
                          k=torch.tensor(saved_diag["k"], dtype=torch.long),
                          y=torch.tensor(saved_diag["y"], dtype=torch.float32),
                          z=torch.tensor(saved_diag["z"], dtype=torch.float32),
                          flow=torch.tensor(saved_diag["flow"], dtype=torch.float32))

    for N in range(completed, target_updates):
        if N in curve_updates:
            evaluate_curves(task, states, base, curve_bank, c, N, output, curves,
                            source_seed_dir)
        if N in diag_updates and not any(int(row["N"]) == N for row in actor_rows):
            new_rows = virtual_actor_update_diagnostic(
                task, states, base, c, data, seed, N, protocol, diag_state, output, device)
            actor_rows.extend(new_rows)
        for method in BRANCHES:
            train_update(task, data, base, states[method], c, seed, N, method,
                         protocol, device, train_logs, grad_samples)
        assert_common_rng(states)
        # Persist only after all four branches finish this same matched update.
        save_suite_state(checkpoint, states, N + 1, config_hash)
        write_json(output / "progress.json", dict(seed=seed,
            completed_actor_updates=N + 1, completed_td_updates=N + 1,
            target_updates=target_updates, updated_at=time.time()))
        pd.DataFrame(train_logs).to_csv(output / "training-log.csv", index=False)
        if (N + 1) % protocol["gradient_log_interval"] == 0:
            pd.DataFrame(grad_samples).to_csv(
                output / "gradient-decomposition-per-sample.csv", index=False)
        if (N + 1) % 10 == 0:
            print("FOUR-BRANCH UPDATE", N + 1, "target", target_updates,
                  flush=True)

    if target_updates in curve_updates:
        evaluate_curves(task, states, base, curve_bank, c,
                        target_updates, output, curves, source_seed_dir)
    if target_updates in diag_updates and not any(int(row["N"]) == target_updates
                                                  for row in actor_rows):
        new_rows = virtual_actor_update_diagnostic(
            task, states, base, c, data, seed, target_updates, protocol,
            diag_state, output, device)
        actor_rows.extend(new_rows)
    save_suite_state(checkpoint, states, target_updates, config_hash)
    pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)
    pd.DataFrame(train_logs).to_csv(output / "training-log.csv", index=False)
    pd.DataFrame(grad_samples).to_csv(output / "gradient-decomposition-per-sample.csv", index=False)
    pd.DataFrame(actor_rows).drop_duplicates(["N", "method"]).sort_values(["N", "method"]).to_csv(
        output / "actor_update_decomposition.csv", index=False)
    return target_updates


def run(output, device=None, continue_to_300=True):
    protocol = json.loads(CONFIG_PATH.read_text())
    if protocol["branches"] != list(BRANCHES) or protocol["seed"] != 59001:
        raise RuntimeError("this runner is intentionally frozen to seed 59001 and four branches")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / "protocol.json").exists():
        raise FileExistsError(f"refusing to use non-empty/unfrozen output {output}")
    if (output / "complete.json").exists():
        print("EXPERIMENT ALREADY COMPLETE", output, flush=True)
        return
    (output / "checkpoints").mkdir(exist_ok=True)
    (c, source, seed_dir, train, valid, base, initial_critic,
     initial_target, states, config_hash) = initialize_run(protocol, output, device)
    task = M.PointObstacle(c)
    diag = diagnostic_batch(train, c, protocol["seed"])
    curve_bank = M.noise_bank(c, protocol["curve_episodes"], M.seed_for(protocol["seed"], 501))
    final_bank = M.noise_bank(c, protocol["final_episodes"], M.seed_for(protocol["seed"], 502))

    completed = run_stage(task, train, base, states, c, protocol["seed"], protocol,
                          output, config_hash, seed_dir,
                          protocol["stage1_actor_updates"], diag, device)
    final_rows = load_rows(output / "final-independent.csv")
    if not any(int(row["N"]) == 100 for row in final_rows):
        if completed > 100:
            raise RuntimeError("N=100 final evaluation missing after N>100 resume; refusing to relabel")
        final_rows = [row for row in final_rows if int(row["N"]) != 100]
        final_rows.extend(evaluate_table(task, states, base, final_bank, 100))
        pd.DataFrame(final_rows).to_csv(output / "final-independent.csv", index=False)
    test50_rows = []
    test50_summaries = []
    test50_path = output / "test50-N100" / "branch-summary.csv"
    if test50_path.exists():
        test50_rows = pd.read_csv(test50_path).to_dict("records")
        counts = pd.read_csv(output / "test50-N100" / "paired-transition-counts.csv").to_dict("records")
        test50_summaries.append(dict(N=100, summary=test50_rows, counts=counts))
    else:
        summary, counts, _ = save_test50(task, states, base, c, protocol, 100,
                                         output, folder_name="test50-N100")
        test50_rows.extend(summary)
        test50_summaries.append(dict(N=100, summary=summary, counts=counts))

    curves = load_rows(output / "curves.csv")
    admission_path = output / "admission.json"
    if admission_path.exists():
        admission = json.loads(admission_path.read_text())
    else:
        admission = admission_decision(final_rows, protocol)
        write_json(admission_path, admission)
    curves = load_rows(output / "curves.csv")
    grad_samples = load_rows(output / "gradient-decomposition-per-sample.csv")
    actor_rows = load_rows(output / "actor_update_decomposition.csv")
    write_report(output, final_rows, curves, admission, grad_samples, actor_rows,
                 test50_summaries, protocol, "stage1-complete")

    if admission["continue_to_300"] and continue_to_300:
        print("ADMISSION PASSED; continuing unchanged four branches to N=300", flush=True)
        stage2 = protocol["stage2_actor_updates"]
        run_stage(task, train, base, states, c, protocol["seed"], protocol,
                  output, config_hash, seed_dir, stage2, diag, device)
        final_rows = [row for row in final_rows if int(row["N"]) != stage2]
        final_rows.extend(evaluate_table(task, states, base, final_bank, stage2))
        pd.DataFrame(final_rows).to_csv(output / "final-independent.csv", index=False)
        summary, counts, _ = save_test50(task, states, base, c, protocol, stage2,
                                         output, folder_name="test50")
        test50_summaries.append(dict(N=stage2, summary=summary, counts=counts))
        curves = load_rows(output / "curves.csv")
        grad_samples = load_rows(output / "gradient-decomposition-per-sample.csv")
        actor_rows = load_rows(output / "actor_update_decomposition.csv")
        write_report(output, final_rows, curves, admission, grad_samples,
                     actor_rows, test50_summaries, protocol, "stage2-complete")
        write_json(output / "complete.json", dict(seed=protocol["seed"],
            final_actor_updates=stage2, admission=admission,
            completed_at=time.time(), status="complete"))
    else:
        if not (output / "test50").exists():
            shutil.copytree(output / "test50-N100", output / "test50")
        write_json(output / "complete.json", dict(seed=protocol["seed"],
            final_actor_updates=100, admission=admission,
            completed_at=time.time(), status="complete_n100_stop"))
    print("EXPERIMENT COMPLETE", output, json.dumps(admission), flush=True)


def make_unique_output():
    return ROOT / "runs" / f"02-qam-point-obstacle-gradient-magnitude-direction-59001-{secrets.token_hex(4)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else make_unique_output()
    try:
        run(output, continue_to_300=True)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", dict(error=repr(exc),
                   traceback=__import__("traceback").format_exc(), time=time.time()))
        raise


if __name__ == "__main__":
    main()
