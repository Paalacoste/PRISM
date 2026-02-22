"""Experiment A.2: Hyperparameter sweep for MetaSR.

Sweeps 4 MetaSR parameters (U_prior, decay, beta, theta_C) over 3 values each,
producing 81 configs x n_runs x 3 phases measurements.

Uses pool initializer pattern: env/mapper/M_star are created once per worker
process, not per task. This avoids pickling the 260x260 M_star matrix 810 times.

Usage:
    python experiments/exp_a_sweep.py [--n_runs 10] [--seed 42] [--workers 7]
                                      [--phase_episodes 200] [--max_ep_steps 500]
                                      [--note "..."]
"""

import csv
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import minigrid  # noqa: F401 -- must import before gym.make
import gymnasium as gym

from prism.config import PRISMConfig, MetaSRConfig
from prism.env.state_mapper import StateMapper
from prism.env.dynamics_wrapper import DynamicsWrapper

from experiments.exp_a_calibration import (
    PRISMExpAAgent,
    train_one_episode,
    measure_calibration,
    generate_goal_positions,
)


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
SWEEP_PARAMS = {
    "U_prior": [0.5, 0.8, 1.0],
    "decay": [0.7, 0.85, 0.95],
    "beta": [5.0, 10.0, 20.0],
    "theta_C": [0.2, 0.3, 0.5],
}


def generate_sweep_configs():
    """Produce 81 MetaSR config dicts (cartesian product of SWEEP_PARAMS)."""
    keys = list(SWEEP_PARAMS.keys())
    values = [SWEEP_PARAMS[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


# ---------------------------------------------------------------------------
# Worker pool initializer + global state
# ---------------------------------------------------------------------------
_w_mapper = None
_w_dyn = None
_w_M_star = None
_w_env_seed = None


def _init_worker(seed, M_star_flat, n_states, max_ep_steps):
    """Called once per worker process. Creates env/mapper, stores M_star."""
    global _w_mapper, _w_dyn, _w_M_star, _w_env_seed

    import minigrid  # noqa: F401
    import gymnasium as gym

    env = gym.make("MiniGrid-FourRooms-v0")
    env.unwrapped.max_steps = max_ep_steps + 100
    env.reset(seed=seed)

    _w_mapper = StateMapper(env)
    _w_dyn = DynamicsWrapper(env, seed=seed)
    _w_M_star = np.frombuffer(M_star_flat).reshape(n_states, n_states).copy()
    _w_env_seed = seed


def _run_single(args):
    """Worker: run 1 config x 1 run x 3 phases using pre-initialized state."""
    (config_dict, config_index, run_idx, seed,
     goal_positions, phase_episodes, max_ep_steps) = args

    state_mapper = _w_mapper
    dyn = _w_dyn
    M_star = _w_M_star
    env_seed = _w_env_seed
    n_states = state_mapper.n_states

    meta_sr_cfg = MetaSRConfig(**config_dict)
    agent_seed = seed + 10000 + run_idx * 100 + config_index

    config = PRISMConfig(meta_sr=meta_sr_cfg)
    agent = PRISMExpAAgent(n_states, state_mapper,
                           config=config, seed=agent_seed)

    goals = [tuple(g) for g in goal_positions]
    phases = []

    for phase_idx in range(3):
        dyn.apply_perturbation("reward_shift",
                               new_goal_pos=goals[phase_idx])

        for _ in range(phase_episodes):
            train_one_episode(dyn, agent, state_mapper,
                              env_seed, max_ep_steps)

        metrics = measure_calibration(agent, M_star)
        phases.append({
            "ece": metrics["ece"],
            "mi": metrics["mi"],
            "hl_stat": metrics["hl_stat"],
            "hl_pvalue": metrics["hl_pvalue"],
            "mean_confidence": metrics["mean_confidence"],
            "mean_error": metrics["mean_error"],
            "mean_uncertainty": metrics["mean_uncertainty"],
            "n_states_visited": metrics["n_states_visited"],
        })

    return (config_dict, config_index, run_idx, phases)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def _save_sweep_results(all_results, output_dir):
    """Save sweep_results.csv (one row per config x run x phase)."""
    csv_path = output_dir / "sweep_results.csv"
    fieldnames = [
        "U_prior", "decay", "beta", "theta_C",
        "run", "phase",
        "ece", "mi", "hl_stat", "hl_pvalue",
        "mean_confidence", "mean_error", "mean_uncertainty",
        "n_states_visited",
    ]

    all_results.sort(key=lambda x: (x[1], x[2]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config_dict, _ci, run_idx, phases in all_results:
            for phase_idx, metrics in enumerate(phases):
                row = {
                    "U_prior": config_dict["U_prior"],
                    "decay": config_dict["decay"],
                    "beta": config_dict["beta"],
                    "theta_C": config_dict["theta_C"],
                    "run": run_idx,
                    "phase": phase_idx + 1,
                }
                for k in ["ece", "mi", "hl_stat", "hl_pvalue",
                          "mean_confidence", "mean_error",
                          "mean_uncertainty"]:
                    row[k] = f"{metrics[k]:.6f}"
                row["n_states_visited"] = metrics["n_states_visited"]
                writer.writerow(row)

    print(f"Results saved to {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
def _print_sweep_summary(all_results):
    """Print top 10 configs, physics alerts, stability check."""
    from collections import defaultdict
    by_config = defaultdict(list)
    for config_dict, ci, run_idx, phases in all_results:
        by_config[ci].append((config_dict, phases))

    config_stats = []
    for ci, entries in by_config.items():
        config_dict = entries[0][0]
        phase3_eces = [phases[2]["ece"] for _, phases in entries]
        phase3_mis = [phases[2]["mi"] for _, phases in entries]

        ece_arr = np.array(phase3_eces)
        mi_arr = np.array(phase3_mis)
        config_stats.append({
            **config_dict,
            "ece_median": float(np.median(ece_arr)),
            "ece_std": float(np.std(ece_arr)),
            "mi_median": float(np.median(mi_arr)),
            "mi_std": float(np.std(mi_arr)),
            "ece_cv": float(np.std(ece_arr) / max(np.mean(ece_arr), 1e-9)),
        })

    config_stats.sort(key=lambda x: x["ece_median"])

    print("\n" + "=" * 90)
    print("SWEEP SUMMARY -- Experiment A.2: MetaSR Hyperparameter Sweep")
    print("=" * 90)

    print(f"\n{'Rank':<5} {'U_prior':>8} {'decay':>6} {'beta':>6} {'theta_C':>8}"
          f"  {'ECE_med':>8} {'ECE_std':>8} {'MI_med':>7} {'MI_std':>7}")
    print("-" * 75)
    for i, cs in enumerate(config_stats[:10]):
        print(f"{i+1:<5} {cs['U_prior']:>8.2f} {cs['decay']:>6.2f}"
              f" {cs['beta']:>6.1f} {cs['theta_C']:>8.2f}"
              f"  {cs['ece_median']:>8.4f} {cs['ece_std']:>8.4f}"
              f" {cs['mi_median']:>7.4f} {cs['mi_std']:>7.4f}")

    best = config_stats[0]
    print(f"\nBest config: U_prior={best['U_prior']}, decay={best['decay']},"
          f" beta={best['beta']}, theta_C={best['theta_C']}")
    print(f"  ECE = {best['ece_median']:.4f} +/- {best['ece_std']:.4f}")
    print(f"  MI  = {best['mi_median']:.4f} +/- {best['mi_std']:.4f}")

    print("\n" + "-" * 60)
    print("PHYSICS ALERTS:")
    alerts = []
    if best["beta"] > 30:
        alerts.append(f"  beta = {best['beta']} -> sigmoid quasi-binary")
    if best["beta"] < 3:
        alerts.append(f"  beta = {best['beta']} -> sigmoid too flat")
    if best["decay"] > 0.97:
        alerts.append(f"  decay = {best['decay']} -> prior too slow")
    if best["decay"] < 0.5:
        alerts.append(f"  decay = {best['decay']} -> prior too fast")
    if best["U_prior"] < 0.3:
        alerts.append(f"  U_prior = {best['U_prior']} -> unvisited low-uncertainty")
    if best["theta_C"] > 0.7:
        alerts.append(f"  theta_C = {best['theta_C']} -> everything high confidence")

    if alerts:
        for a in alerts:
            print(a)
    else:
        print("  No alerts -- parameters are in reasonable ranges")

    print("\n" + "-" * 60)
    print("STABILITY (best config):")
    print(f"  ECE CV: {best['ece_cv']:.3f}"
          f"  ({'stable' if best['ece_cv'] < 0.3 else 'UNSTABLE' if best['ece_cv'] > 0.5 else 'marginal'})")

    print("\n" + "-" * 60)
    print("ECE <-> MI CONCORDANCE:")
    top10_ece = [cs["ece_median"] for cs in config_stats[:10]]
    top10_mi = [cs["mi_median"] for cs in config_stats[:10]]
    from scipy.stats import spearmanr
    rho, p = spearmanr(top10_ece, top10_mi)
    print(f"  Spearman(ECE, MI) in top-10: rho={rho:.3f}, p={p:.4f}")
    if rho < -0.3:
        print("  Concordant (lower ECE -> higher MI)")
    elif rho > 0.3:
        print("  WARNING: Divergent")
    else:
        print("  Weak relationship")

    defaults = {"U_prior": 0.8, "decay": 0.85, "beta": 10.0, "theta_C": 0.3}
    for i, cs in enumerate(config_stats):
        if (cs["U_prior"] == defaults["U_prior"]
                and cs["decay"] == defaults["decay"]
                and cs["beta"] == defaults["beta"]
                and cs["theta_C"] == defaults["theta_C"]):
            print(f"\nDefault config (0.8, 0.85, 10.0, 0.3) ranks #{i+1}/81"
                  f"  ECE={cs['ece_median']:.4f}, MI={cs['mi_median']:.4f}")
            break
    else:
        print("\nWARNING: Default config not found in sweep results")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_sweep(n_runs=10, seed=42, n_workers=None, note="",
              phase_episodes=200, max_ep_steps=500):
    """Run the full MetaSR hyperparameter sweep.

    Uses pool initializer: env/mapper/M_star created once per worker (not per
    task). Fine-grained tasks: 1 task = 1 config x 1 run.
    """
    if n_workers is None:
        n_workers = max(os.cpu_count() - 1, 1)

    run_timestamp = datetime.now()
    run_name = f"run_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("results/sweep") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # --- Setup: env, mapper, M* (once in main process) ---
    print("Setting up environment...")
    env = gym.make("MiniGrid-FourRooms-v0")
    env.unwrapped.max_steps = max_ep_steps + 100
    env.reset(seed=seed)

    state_mapper = StateMapper(env)
    n_states = state_mapper.n_states
    print(f"  FourRooms: {n_states} accessible states")

    print("Computing true SR matrix M*...")
    dyn_wrapper = DynamicsWrapper(env)
    T = dyn_wrapper.get_true_transition_matrix(state_mapper)
    gamma = PRISMConfig().sr.gamma
    M_star = np.linalg.inv(np.eye(n_states) - gamma * T)
    print(f"  M* shape: {M_star.shape}, gamma={gamma}")

    # Flatten M_star for sharing via initializer (pickled once per worker)
    M_star_flat = M_star.tobytes()

    # --- Pre-generate goal positions ---
    print("Pre-generating goal positions...")
    goal_positions_list = []
    for run_idx in range(n_runs):
        rng = np.random.default_rng(seed + run_idx)
        positions = generate_goal_positions(state_mapper, rng)
        goal_positions_list.append([list(p) for p in positions])

    # --- Generate sweep configs ---
    configs = generate_sweep_configs()
    n_configs = len(configs)
    n_tasks = n_configs * n_runs
    total_measures = n_configs * n_runs * 3
    print(f"\n{n_configs} configs x {n_runs} runs = {n_tasks} tasks "
          f"({total_measures} measures)")
    print(f"Workers: {n_workers}\n")

    t_start = time.time()

    # --- Build fine-grained worker args (lightweight — no M_star) ---
    worker_args = []
    for i, cfg in enumerate(configs):
        for run_idx in range(n_runs):
            worker_args.append((
                cfg, i, run_idx, seed,
                goal_positions_list[run_idx],
                phase_episodes, max_ep_steps,
            ))

    # --- Execute with progress tracking ---
    progress_path = run_dir / "_progress.json"
    all_results = []

    def _write_progress(completed, total, t_start):
        elapsed = time.time() - t_start
        eta = (elapsed / max(completed, 1)) * (total - completed)
        progress = {
            "completed": completed,
            "total": total,
            "pct": round(100 * completed / total, 1),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
        }
        with open(progress_path, "w") as fp:
            json.dump(progress, fp)

    _write_progress(0, n_tasks, t_start)

    if n_workers == 1:
        # Sequential: init once in-process
        _init_worker(seed, M_star_flat, n_states, max_ep_steps)
        for i, args in enumerate(tqdm(worker_args, desc="Sweep tasks")):
            result = _run_single(args)
            all_results.append(result)
            _write_progress(i + 1, n_tasks, t_start)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(seed, M_star_flat, n_states, max_ep_steps),
        ) as executor:
            futures = {executor.submit(_run_single, args): idx
                       for idx, args in enumerate(worker_args)}
            for i, fut in enumerate(as_completed(futures)):
                all_results.append(fut.result())
                _write_progress(i + 1, n_tasks, t_start)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s "
          f"({elapsed/n_tasks:.2f}s per task, "
          f"{elapsed/n_configs:.1f}s per config)")

    # --- Save ---
    _save_sweep_results(all_results, run_dir)

    # --- Summary ---
    _print_sweep_summary(all_results)

    # --- Run info ---
    run_info = {
        "timestamp": run_timestamp.isoformat(timespec="seconds"),
        "experiment": "sweep_a",
        "seed": seed,
        "n_runs": n_runs,
        "n_configs": n_configs,
        "n_tasks": n_tasks,
        "sweep_params": SWEEP_PARAMS,
        "phase_episodes": phase_episodes,
        "max_ep_steps": max_ep_steps,
        "n_workers": n_workers,
        "total_measures": total_measures,
        "elapsed_seconds": round(elapsed, 1),
        "note": note,
    }
    run_info_path = run_dir / "run_info.json"
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nRun info saved to {run_info_path}")

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment A.2: MetaSR hyperparameter sweep")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Runs per config (default 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count-1)")
    parser.add_argument("--note", type=str, default="",
                        help="Free-text note for this run")
    parser.add_argument("--phase_episodes", type=int, default=200,
                        help="Episodes per phase (default 200)")
    parser.add_argument("--max_ep_steps", type=int, default=500,
                        help="Max steps per episode (default 500)")
    args = parser.parse_args()

    run_sweep(
        n_runs=args.n_runs,
        seed=args.seed,
        n_workers=args.workers,
        note=args.note,
        phase_episodes=args.phase_episodes,
        max_ep_steps=args.max_ep_steps,
    )


if __name__ == "__main__":
    main()
