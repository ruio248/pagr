"""Development selection and fresh-seed Lean AM comparison for pathwise TC-Robust QAM."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
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
sys.path.insert(0, str(ROOT / "scripts"))
import terminal_consistent_robust_qam as robust

spec = importlib.util.spec_from_file_location(
    "point_obstacle_terminal_consistent_runner",
    ROOT / "notebooks" / "02_qam_point_obstacle.py")
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)
torch.set_num_threads(min(4, torch.get_num_threads()))

CONFIG_PATH = ROOT / "configs" / "terminal_consistent_robust_qam.json"


def read_protocol():
    return json.loads(CONFIG_PATH.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def effective_config(p):
    c, _ = M.load_config()
    return replace(c, actor_updates=p["formal_updates"],
                   curve_updates=tuple(p["checkpoints"]),
                   curve_episodes=p["curve_episodes"],
                   final_episodes=p["final_episodes"])


def source_manifest(c, p):
    names = (
        "configs/qam_point_obstacle.json",
        "configs/terminal_consistent_robust_qam.json",
        "notebooks/02_qam_point_obstacle.py",
        "notebooks/02_qam_td_terminal_error.py",
        "scripts/terminal_consistent_robust_qam.py",
        "scripts/run_terminal_consistent_robust_qam.py",
    )
    return dict(protocol=p, config=asdict(c), source_sha256={name: digest(ROOT / name)
                                                             for name in names})


def create_run(stage, p, c):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = ROOT / "runs" / f"02-qam-point-obstacle-terminal-consistent-robust-{stage}-{stamp}-{secrets.token_hex(2)}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen-protocol.json", source_manifest(c, p))
    return output


def load_or_create_initial(c, seed, folder):
    """One deterministic dataset/Base/Critic initialization shared by branches."""
    folder.mkdir(parents=True, exist_ok=True)
    init = folder / "shared-initialization.pt"
    train_path, valid_path = folder / "train-transitions.npz", folder / "valid-transitions.npz"
    if not init.exists():
        task = M.PointObstacle(c)
        train_raw, train_summary = M.collect_episodes(task, c.train_episodes,
                                                       M.seed_for(seed, 1))
        valid_raw, valid_summary = M.collect_episodes(task, c.valid_episodes,
                                                       M.seed_for(seed, 2))
        np.savez_compressed(train_path, **train_raw)
        np.savez_compressed(valid_path, **valid_raw)
        write_json(folder / "data-summary.json", dict(train=train_summary,
                                                        valid=valid_summary))
        train = M.as_torch(train_raw)
        base, critic, target = M.pretrain(task, train, c, seed, folder)
        # M.pretrain writes the exact shared initialization expected by QAM.
        del base, critic, target
    if not valid_path.exists():
        raise FileNotFoundError(valid_path)
    train_archive = np.load(train_path)
    train = M.as_torch({name: train_archive[name] for name in train_archive.files})
    saved = torch.load(init, map_location="cpu", weights_only=True)
    base = M.core.ConditionalVelocity(c)
    base.load_state_dict(saved["base"])
    critic = M.core.Critic(c)
    critic.load_state_dict(saved["critic"])
    target = M.core.Critic(c)
    target.load_state_dict(saved["target"])
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return train, base, critic, target


def critic_gradient_scale(critic, valid, count):
    """Freeze an error-radius scale using only the initialization Critic."""
    n = min(count, len(valid["k"]))
    ids = torch.linspace(0, len(valid["k"]) - 1, n).round().long()
    k, y, action = valid["k"][ids], valid["y"][ids], valid["a"][ids]
    with torch.enable_grad():
        a = action.detach().requires_grad_(True)
        q = critic(k, y, a).mean(-1)
        grad = torch.autograd.grad(q.sum(), a)[0].detach()
    scale = float(grad.norm(dim=-1).median())
    if not np.isfinite(scale) or scale <= 1e-10:
        raise ValueError(f"invalid initialization gradient scale: {scale}")
    return scale


def new_state(base, initial_critic, initial_target, c, seed):
    actor = copy.deepcopy(base)
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    critic, target = copy.deepcopy(initial_critic), copy.deepcopy(initial_target)
    return dict(actor=actor, critic=critic, target=target,
                aopt=torch.optim.Adam(actor.parameters(), lr=c.actor_lr),
                copt=torch.optim.Adam(critic.parameters(), lr=c.critic_lr),
                rng=M.core.generator(M.seed_for(seed, 701)))


def save_checkpoint(folder, method, N, state, epsilon, scale):
    torch.save(dict(N=N, method=method, epsilon=epsilon,
                    epsilon_scale=scale, actor=state["actor"].state_dict(),
                    critic=state["critic"].state_dict(), target=state["target"].state_dict(),
                    actor_optimizer=state["aopt"].state_dict(),
                    critic_optimizer=state["copt"].state_dict(),
                    rng=state["rng"].get_state()), folder / f"{method}-N{N}.pt")


def train_branch(task, train, base, initial_critic, initial_target, c, seed,
                 method, epsilon, epsilon_scale, updates, checkpoints, bank,
                 folder, save_all=True):
    folder.mkdir(parents=True, exist_ok=True)
    state = new_state(base, initial_critic, initial_target, c, seed)
    actor, critic, target = state["actor"], state["critic"], state["target"]
    aopt, copt, rng = state["aopt"], state["copt"], state["rng"]
    curves, logs = [], []
    for N in range(updates + 1):
        if N in checkpoints:
            stats = M.evaluate(task, actor, base, bank)
            curves.append(dict(seed=seed, method=method, N=N,
                               epsilon=epsilon, epsilon_scale=epsilon_scale, **stats))
            if save_all:
                save_checkpoint(folder, method, N, state, epsilon, epsilon_scale)
            print("CURVE", seed, method, N, json.dumps(stats), flush=True)
        if N == updates:
            break
        idx = torch.randint(len(train["k"]), (c.critic_batch,), generator=rng)
        td_loss, q_mean = M.td_step(critic, target, copt, train, idx, rng,
                                    c, actor, base)
        idx = torch.randint(len(train["k"]), (c.actor_batch,), generator=rng)
        k, y = train["k"][idx], train["y"][idx]
        z = torch.randn((len(idx), c.dimension), generator=rng)
        flow = torch.randn((len(idx), c.flow_steps, c.dimension), generator=rng)
        if method == "lean_am":
            info = M.core.actor_step(actor, aopt, base, critic, task, c,
                                     k, y, z, flow, "learned")
        elif method == "tc_robust":
            info = robust.terminal_consistent_actor_step(
                actor, aopt, base, critic, task, c, k, y, z, flow,
                epsilon, M.core)
        else:
            raise ValueError(method)
        if (N + 1) % 10 == 0:
            logs.append(dict(seed=seed, method=method, N=N + 1,
                             td_loss=td_loss, q_mean=q_mean, epsilon=epsilon,
                             epsilon_scale=epsilon_scale, **info))
    pd.DataFrame(curves).to_csv(folder / f"{method}-curves.csv", index=False)
    pd.DataFrame(logs).to_csv(folder / f"{method}-train.csv", index=False)
    return state, curves, logs


def dev(output):
    p = read_protocol()
    c = effective_config(p)
    c = replace(c, actor_updates=p["development_updates"],
                curve_updates=(p["development_updates"],),
                curve_episodes=p["development_episodes"],
                final_episodes=p["development_episodes"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen-protocol.json", source_manifest(c, p))
    seed = p["development_seed"]
    folder = output / f"seed-{seed}"
    train, base, initial_critic, initial_target = load_or_create_initial(c, seed, folder)
    valid_archive = np.load(folder / "valid-transitions.npz")
    valid = M.as_torch({name: valid_archive[name] for name in valid_archive.files})
    scale = critic_gradient_scale(initial_critic, valid,
                                  p["gradient_scale_validation_points"])
    task = M.PointObstacle(c)
    bank = M.noise_bank(c, p["development_episodes"], M.seed_for(seed, 1502))
    results = []
    candidates = [("lean_am", 0.)] + [("tc_robust", x) for x in p["epsilon_scales"] if x > 0]
    for method, ratio in candidates:
        epsilon = ratio * scale
        name = "lean_am" if ratio == 0 else f"epsilon_scale_{str(ratio).replace('.', 'p')}"
        state, curves, _ = train_branch(task, train, base, initial_critic,
            initial_target, c, seed, method, epsilon, ratio,
            p["development_updates"], (0, p["development_updates"]), bank,
            folder / name, save_all=False)
        result = dict(curves[-1])
        result.update(epsilon_scale=ratio, epsilon=epsilon,
                      initialization_gradient_scale=scale)
        results.append(result)
        pd.DataFrame(results).to_csv(output / "development-sweep.csv", index=False)
    order = sorted(results, key=lambda r: (-r["success_rate"], r["collision_rate"],
                                           -r["discounted_return"], r["epsilon_scale"]))
    selected = order[0]
    write_json(output / "selected-radius.json", dict(
        epsilon_scale=selected["epsilon_scale"], epsilon=selected["epsilon"],
        initialization_gradient_scale=scale,
        selection_rule=p["epsilon_selection"],
        development_seed=seed, development_updates=p["development_updates"],
        results=results))
    print("RADIUS SELECTED", json.dumps({"epsilon_scale": selected["epsilon_scale"],
                                        "epsilon": selected["epsilon"]}), flush=True)


def paired_test(task, states, base, c, seed, output, p):
    bank = M.noise_bank(c, p["paired_episodes"], M.seed_for(seed, 9502))
    records = {}
    for method, state in states.items():
        stats, _, status = M.evaluate(task, state["actor"], base, bank, trajectories=True)
        records[method] = (stats, status)
    ordinary = records["lean_am"][1]
    robust_status = records["tc_robust"][1]
    paired = pd.DataFrame(dict(seed=seed, episode=np.arange(len(ordinary)),
                               lean_am_status=ordinary,
                               tc_robust_status=robust_status))
    paired.to_csv(output / "paired-outcomes.csv", index=False)
    return dict(seed=seed,
        lean_failure_to_robust_success=int(((ordinary != "success") &
                                            (robust_status == "success")).sum()),
        lean_success_to_robust_failure=int(((ordinary == "success") &
                                            (robust_status != "success")).sum()),
        both_success=int(((ordinary == "success") & (robust_status == "success")).sum()),
        both_failure=int(((ordinary != "success") & (robust_status != "success")).sum()))


def summarize(output, all_curves, all_final, all_pairs, p, selected):
    curves = pd.DataFrame(all_curves)
    final = pd.DataFrame(all_final)
    curves.to_csv(output / "formal-curves.csv", index=False)
    final.to_csv(output / "formal-final.csv", index=False)
    pd.DataFrame(all_pairs).to_csv(output / "paired-summary.csv", index=False)
    aggregate = final.groupby("method").agg(
        success_mean=("success_rate", "mean"), success_sd=("success_rate", "std"),
        collision_mean=("collision_rate", "mean"), collision_sd=("collision_rate", "std"),
        return_mean=("discounted_return", "mean"), return_sd=("discounted_return", "std"))
    aggregate.to_csv(output / "formal-aggregate.csv")
    wide = final.pivot(index="seed", columns="method", values="success_rate")
    differences = (wide["tc_robust"] - wide["lean_am"]) * 100.
    consistency = dict(seed_differences_pp={str(int(k)): float(v) for k, v in differences.items()},
                       mean_difference_pp=float(differences.mean()),
                       seed_sd_pp=float(differences.std(ddof=1)),
                       robust_wins=int((differences > 0).sum()),
                       ties=int((differences == 0).sum()),
                       robust_losses=int((differences < 0).sum()))
    write_json(output / "paired-success-differences.json", consistency)
    write_json(output / "paired-outcomes-summary.json", dict(per_seed=all_pairs,
        aggregate={key: sum(row[key] for row in all_pairs) for key in all_pairs[0]
                   if key != "seed"}))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for method, group in curves.groupby("method"):
        summary_curve = group.groupby("N")["success_rate"].agg(["mean", "std"])
        color = "#2676b8" if method == "lean_am" else "#d2632a"
        axes[0].plot(summary_curve.index, 100 * summary_curve["mean"],
                     marker="o", label=method, color=color)
        axes[0].fill_between(summary_curve.index,
            100 * (summary_curve["mean"] - summary_curve["std"].fillna(0)),
            100 * (summary_curve["mean"] + summary_curve["std"].fillna(0)),
            color=color, alpha=.15)
    axes[0].set(xlabel="Actor updates", ylabel="Success rate (%)",
                title="Matched training budget")
    axes[0].legend(frameon=False)
    for i, row in enumerate(aggregate.itertuples()):
        axes[1].bar(i, 100 * row.success_mean,
                    yerr=100 * row.success_sd, capsize=4,
                    color="#2676b8" if row.Index == "lean_am" else "#d2632a")
    axes[1].set_xticks(range(len(aggregate)), aggregate.index)
    axes[1].set(ylabel="Success rate (%)", title="Independent final evaluation")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Terminal-consistent pathwise robust QAM: exploratory result",
        "",
        "This experiment compares the ordinary learned-Q Lean AM update with a frozen-path robust Hamiltonian target. A single terminal perturbation is shared across all Flow steps of each trajectory. The perturbation may differ across trajectories, so this is not a globally coherent delta(s,a1) field.",
        "",
        f"The new-seed comparison uses epsilon scale **{selected['epsilon_scale']}**. The development selector's best overall choice was scale **{selected['development_selected_epsilon_scale']}**; the positive-radius run is therefore a sensitivity check, not the development-selected winner. Epsilon is this scale times each seed's initialization-Critic gradient scale. Development used seed {p['development_seed']} at {p['development_updates']} updates; evaluation seeds were not used to choose the radius.",
        "",
        f"The formal comparison uses seeds {', '.join(map(str, p['formal_seeds']))}, {p['formal_updates']} Actor updates per branch, shared per-seed initialization and TD/Actor update budgets, and {p['final_episodes']} independent final episodes per seed.",
        "",
        "## Final results",
        "",
        "| Method | Success (%) | Collision (%) | Discounted return |",
        "|---|---:|---:|---:|",
    ]
    for method in ("lean_am", "tc_robust"):
        row = aggregate.loc[method]
        lines.append(f"| {method} | {100*row.success_mean:.2f} ± {100*row.success_sd:.2f} | {100*row.collision_mean:.2f} ± {100*row.collision_sd:.2f} | {row.return_mean:.4f} ± {row.return_sd:.4f} |")
    lines.extend(["", f"Paired robust-minus-Lean-AM success difference: **{consistency['mean_difference_pp']:+.2f} pp** across three seeds (sample SD {consistency['seed_sd_pp']:.2f} pp; {consistency['robust_wins']} wins, {consistency['ties']} ties, {consistency['robust_losses']} losses).",
        "",
        "These are exploratory results on the existing 2D obstacle toy and the current finite-chain QAM implementation. They test whether the candidate helps under this training protocol; they do not establish a general robust-control guarantee or a coherent state-action gradient-error field. The ordinary and robust branches share initialization, minibatch/noise streams at the start, and evaluation banks; their later TD critics naturally diverge as their Actors diverge.",
        "",
        "![Training and final comparison](comparison.png)",
        "",
        "Full curves, per-seed finals, paired episodes, checkpoints, and the frozen protocol are stored in this run directory.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def formal(output, epsilon_override=None):
    p = read_protocol()
    c = effective_config(p)
    development_selection = json.loads((output / "selected-radius.json").read_text())
    ratio = (development_selection["epsilon_scale"] if epsilon_override is None
             else float(epsilon_override))
    allowed = [0.] + list(p["epsilon_scales"])
    if ratio not in allowed:
        raise ValueError(f"epsilon scale {ratio} not in frozen candidate set {allowed}")
    selection = dict(development_selection,
        development_selected_epsilon_scale=development_selection["epsilon_scale"],
        epsilon_scale=ratio,
        radius_choice=("development winner" if epsilon_override is None else
                       "explicit frozen-candidate sensitivity check"))
    output = output / "formal"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frozen-protocol.json", source_manifest(c, p))
    write_json(output / "formal-radius-choice.json", selection)
    all_curves, all_final, all_pairs = [], [], []
    for seed in p["formal_seeds"]:
        folder = output / f"seed-{seed}"
        train, base, initial_critic, initial_target = load_or_create_initial(c, seed, folder)
        valid_archive = np.load(folder / "valid-transitions.npz")
        valid = M.as_torch({name: valid_archive[name] for name in valid_archive.files})
        scale = critic_gradient_scale(initial_critic, valid,
                                      p["gradient_scale_validation_points"])
        epsilon = ratio * scale
        task = M.PointObstacle(c)
        curve_bank = M.noise_bank(c, p["curve_episodes"], M.seed_for(seed, 501))
        final_bank = M.noise_bank(c, p["final_episodes"], M.seed_for(seed, 502))
        states = {}
        for method in ("lean_am", "tc_robust"):
            state, curves, _ = train_branch(task, train, base, initial_critic,
                initial_target, c, seed, method, epsilon if method == "tc_robust" else 0.,
                ratio if method == "tc_robust" else 0., c.actor_updates,
                c.curve_updates, curve_bank, folder / method)
            states[method] = state
            all_curves.extend(curves)
            stats = M.evaluate(task, state["actor"], base, final_bank)
            all_final.append(dict(seed=seed, method=method, N=c.actor_updates,
                                  epsilon=epsilon if method == "tc_robust" else 0.,
                                  epsilon_scale=ratio if method == "tc_robust" else 0.,
                                  initialization_gradient_scale=scale, **stats))
            pd.DataFrame(all_curves).to_csv(output / "formal-curves.csv", index=False)
            pd.DataFrame(all_final).to_csv(output / "formal-final.csv", index=False)
        pair = paired_test(task, states, base, c, seed, folder, p)
        all_pairs.append(pair)
        write_json(folder / "complete.json", dict(seed=seed, epsilon_scale=ratio,
            epsilon=epsilon, final=[row for row in all_final if row["seed"] == seed],
            paired=pair, completed_at=time.time()))
        print("SEED COMPLETE", seed, json.dumps(pair), flush=True)
    summarize(output, all_curves, all_final, all_pairs, p, selection)
    print("FORMAL COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dev", "formal"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epsilon-scale", type=float,
                        help="optional frozen candidate override for an explicitly exploratory sensitivity run")
    args = parser.parse_args()
    if args.stage == "dev":
        output = args.output or create_run("dev", read_protocol(), effective_config(read_protocol()))
        dev(output)
        print("DEV OUTPUT", output, flush=True)
    else:
        if args.output is None:
            raise ValueError("formal requires --output pointing to the completed development run")
        formal(args.output, args.epsilon_scale)


if __name__ == "__main__":
    main()
