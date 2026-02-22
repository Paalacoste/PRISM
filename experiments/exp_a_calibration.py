"""Experiment A: Calibration and iso-structurality (tests P1 + P2).

Protocol:
    - FourRooms 19x19, 260 accessible states
    - 3 phases of 200 episodes each:
      Phase 1: stable learning (goal at position A)
      Phase 2: reward_shift perturbation (goal moves to B)
      Phase 3: reward_shift perturbation (goal moves to C)
    - 5 conditions: PRISM, SR-Global, SR-Count, SR-Bayesian, Random-Conf
    - 100 runs per condition, metrics measured at end of each phase
    - Output: calibration_results.csv, u_snapshots/, reliability/, run_info.json

Uses pool initializer pattern: env/mapper/M_star are created once per worker
process, not per task.  Fine-grained tasks: 1 task = 1 (condition, run_idx).

Usage:
    python -m experiments.exp_a_calibration [--n_runs 100] [--seed 42] [--workers 4]
        [--decay 0.95] [--beta 5] [--U_prior 0.8] [--theta_C 0.3]
"""

import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import minigrid  # noqa: F401 — must import before gym.make
import gymnasium as gym

from prism.agent.sr_layer import SRLayer
from prism.agent.meta_sr import MetaSR
from prism.agent.controller import PRISMController
from prism.config import PRISMConfig, MetaSRConfig
from prism.env.state_mapper import StateMapper
from prism.env.dynamics_wrapper import DynamicsWrapper
from prism.env.exploration_task import place_goals, get_room_cells
from prism.baselines.calibration_baselines import (
    SRGlobalConf,
    SRCountConf,
    SRBayesianConf,
    RandomConfAgent,
)
from prism.analysis.calibration import (
    sr_errors,
    sr_accuracies,
    expected_calibration_error,
    hosmer_lemeshow_test,
    metacognitive_index,
    reliability_diagram_data,
)
from prism.analysis.metrics import bootstrap_ci, compare_conditions


# MiniGrid movement actions
MOVEMENT_ACTIONS = [0, 1, 2]

CONDITIONS = [
    "PRISM",
    "SR-Global",
    "SR-Count",
    "SR-Bayesian",
    "Random-Conf",
]

PHASE_EPISODES = 200

CSV_FIELDNAMES = [
    "condition", "run", "phase", "ece", "mi", "hl_stat", "hl_pvalue",
    "mean_confidence", "mean_error", "mean_uncertainty", "n_states_visited",
]


# ---------------------------------------------------------------------------
# PRISM adapter for Exp A
# ---------------------------------------------------------------------------
class PRISMExpAAgent:
    """Adapts PRISM components (SR + MetaSR + Controller) for Exp A loop."""

    def __init__(self, n_states, state_mapper, config=None, seed=42):
        if config is None:
            config = PRISMConfig()

        self.n_states = n_states
        self.mapper = state_mapper
        self.sr = SRLayer(
            n_states, config.sr.gamma, config.sr.alpha_M, config.sr.alpha_R,
        )
        self.meta_sr = MetaSR(
            n_states,
            buffer_size=config.meta_sr.buffer_size,
            U_prior=config.meta_sr.U_prior,
            decay=config.meta_sr.decay,
            beta=config.meta_sr.beta,
            theta_C=config.meta_sr.theta_C,
            theta_change=config.meta_sr.theta_change,
        )
        self.controller = PRISMController(
            self.sr, self.meta_sr, state_mapper,
            epsilon_min=config.controller.epsilon_min,
            epsilon_max=config.controller.epsilon_max,
            lambda_explore=config.controller.lambda_explore,
            theta_idk=config.controller.theta_idk,
        )
        self.controller.seed(seed)
        self.visit_counts = np.zeros(n_states, dtype=np.int64)
        self.total_steps = 0

    def select_action(self, s, available_actions, agent_dir):
        action, _, _ = self.controller.select_action(
            s, available_actions, agent_dir=agent_dir,
        )
        return action

    def update(self, s, s_next, reward):
        delta_M = self.sr.update(s, s_next, reward)
        self.meta_sr.observe(s, delta_M)
        self.visit_counts[s] += 1
        self.total_steps += 1

    def confidence(self, s):
        return self.meta_sr.confidence(s)

    def all_confidences(self):
        return self.meta_sr.all_confidences()

    def all_uncertainties(self):
        return self.meta_sr.all_uncertainties()

    def exploration_bonus(self, s):
        return self.meta_sr.uncertainty(s)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def make_agent(condition, n_states, state_mapper, seed, config=None):
    """Create an agent for the given condition.

    Args:
        config: PRISMConfig instance (only used for PRISM condition).
    """
    common = dict(n_states=n_states, state_mapper=state_mapper, seed=seed)
    if condition == "PRISM":
        return PRISMExpAAgent(config=config, **common)
    elif condition == "SR-Global":
        return SRGlobalConf(epsilon=0.1, **common)
    elif condition == "SR-Count":
        return SRCountConf(epsilon=0.1, **common)
    elif condition == "SR-Bayesian":
        return SRBayesianConf(epsilon=0.1, **common)
    elif condition == "Random-Conf":
        return RandomConfAgent(**common)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Single-episode training
# ---------------------------------------------------------------------------
def train_one_episode(env, agent, state_mapper, env_seed, max_ep_steps=500):
    """Run one training episode. Returns set of visited state indices."""
    obs, _ = env.reset(seed=env_seed)
    env.unwrapped.max_steps = max_ep_steps + 100
    s = state_mapper.get_index(tuple(env.unwrapped.agent_pos))
    visited = {s}
    done = False
    step = 0

    while not done and step < max_ep_steps:
        agent_dir = env.unwrapped.agent_dir
        action = agent.select_action(s, MOVEMENT_ACTIONS, agent_dir)
        obs, reward, terminated, truncated, _ = env.step(action)
        s_next = state_mapper.get_index(tuple(env.unwrapped.agent_pos))
        agent.update(s, s_next, reward)
        s = s_next
        visited.add(s)
        step += 1
        done = terminated or truncated

    return visited


# ---------------------------------------------------------------------------
# Calibration measurement
# ---------------------------------------------------------------------------
def measure_calibration(agent, M_star):
    """Compute all calibration metrics for the current agent state.

    Returns dict with CSV columns + internal arrays for snapshots.
    """
    M = agent.sr.M

    errors = sr_errors(M, M_star)
    accuracies = sr_accuracies(M, M_star, percentile=50)
    confidences = agent.all_confidences()
    uncertainties = agent.all_uncertainties()

    ece = expected_calibration_error(confidences, accuracies)
    hl_stat, hl_pvalue = hosmer_lemeshow_test(confidences, accuracies)
    mi_rho, _ = metacognitive_index(uncertainties, M, M_star)
    reliability = reliability_diagram_data(confidences, accuracies)

    n_visited = int(np.sum(agent.visit_counts > 0))

    return {
        "ece": ece,
        "mi": mi_rho,
        "hl_stat": hl_stat,
        "hl_pvalue": hl_pvalue,
        "mean_confidence": float(np.mean(confidences)),
        "mean_error": float(np.mean(errors)),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "n_states_visited": n_visited,
        # Internal (not saved to CSV):
        "_uncertainties": uncertainties,
        "_reliability": reliability,
    }


# ---------------------------------------------------------------------------
# Goal position generation
# ---------------------------------------------------------------------------
def generate_goal_positions(state_mapper, rng):
    """Generate 3 goal positions in 3 different rooms.

    Returns list of 3 (x, y) tuples.
    """
    rooms = get_room_cells(state_mapper)

    # Shuffle room order so each run gets different assignment
    room_indices = list(range(len(rooms)))
    rng.shuffle(room_indices)

    positions = []
    for i in range(3):
        room_idx = room_indices[i % len(rooms)]
        cell_idx = rng.choice(rooms[room_idx])
        positions.append(state_mapper.get_pos(cell_idx))

    return positions


# ---------------------------------------------------------------------------
# Reliability data aggregation
# ---------------------------------------------------------------------------
def aggregate_reliability(reliability_list, n_bins=10):
    """Average reliability data across runs (weighted by bin counts)."""
    agg = {
        "bin_confidences": [0.0] * n_bins,
        "bin_accuracies": [0.0] * n_bins,
        "bin_counts": [0] * n_bins,
        "bin_centers": reliability_list[0]["bin_centers"],
    }
    for rel in reliability_list:
        for i in range(n_bins):
            count = rel["bin_counts"][i]
            agg["bin_counts"][i] += count
            agg["bin_confidences"][i] += rel["bin_confidences"][i] * count
            agg["bin_accuracies"][i] += rel["bin_accuracies"][i] * count

    for i in range(n_bins):
        if agg["bin_counts"][i] > 0:
            agg["bin_confidences"][i] /= agg["bin_counts"][i]
            agg["bin_accuracies"][i] /= agg["bin_counts"][i]

    return agg


# ---------------------------------------------------------------------------
# Resume / incremental-save helpers
# ---------------------------------------------------------------------------
def _load_completed_tasks(csv_path):
    """Load completed (condition, run_idx) pairs from an existing CSV.

    A task is complete only when all 3 phases are present.
    Malformed lines (e.g. from a crash mid-write) are silently skipped.
    """
    if not csv_path.exists():
        return set()

    counts = defaultdict(set)
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (row["condition"], int(row["run"]))
                    counts[key].add(int(row["phase"]))
                except (KeyError, ValueError):
                    continue
    except Exception:
        return set()

    return {key for key, phases in counts.items() if phases == {1, 2, 3}}


def _append_csv_rows(csv_path, run_phases):
    """Append 3 phase rows for one completed task, with fsync for safety."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for metrics in run_phases:
            row = {k: metrics[k] for k in CSV_FIELDNAMES}
            for k in ["ece", "mi", "hl_stat", "hl_pvalue",
                      "mean_confidence", "mean_error", "mean_uncertainty"]:
                row[k] = f"{row[k]:.6f}"
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _save_u_snapshot_if_run0(cond_name, run_idx, run_phases, u_dir,
                             state_mapper):
    """Save U-snapshot .npy files immediately when run_idx == 0."""
    if run_idx != 0:
        return
    u_dir.mkdir(exist_ok=True)
    for metrics in run_phases:
        phase = metrics["phase"]
        U = metrics["_uncertainties"]
        U_grid = state_mapper.to_grid(U)
        np.save(u_dir / f"{cond_name}_U_phase{phase}.npy", U_grid)


def _rewrite_csv_sorted(csv_path, conditions):
    """Re-read the CSV, sort rows by (condition, run, phase), rewrite."""
    if not csv_path.exists():
        return

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    cond_order = {c: i for i, c in enumerate(conditions)}
    rows.sort(key=lambda r: (
        cond_order.get(r["condition"], 999),
        int(r["run"]),
        int(r["phase"]),
    ))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _load_results_from_csv(csv_path, conditions, n_runs):
    """Rebuild an all_results dict from the CSV (for summary / statistics).

    Returns {condition: [run_phases_0, ..., run_phases_{n-1}]}
    where each run_phases is a list of 3 metric dicts.
    Runs without all 3 phases are dropped.
    """
    grouped = defaultdict(lambda: defaultdict(list))

    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            run_idx = int(row["run"])
            metrics = {
                "condition": cond,
                "run": run_idx,
                "phase": int(row["phase"]),
                "ece": float(row["ece"]),
                "mi": float(row["mi"]),
                "hl_stat": float(row["hl_stat"]),
                "hl_pvalue": float(row["hl_pvalue"]),
                "mean_confidence": float(row["mean_confidence"]),
                "mean_error": float(row["mean_error"]),
                "mean_uncertainty": float(row["mean_uncertainty"]),
                "n_states_visited": int(row["n_states_visited"]),
            }
            grouped[cond][run_idx].append(metrics)

    all_results = {}
    for cond in conditions:
        if cond not in grouped:
            continue
        runs = []
        for run_idx in sorted(grouped[cond]):
            phases = grouped[cond][run_idx]
            if len(phases) == 3:
                phases.sort(key=lambda m: m["phase"])
                runs.append(phases)
        all_results[cond] = runs

    return all_results


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

    # Disable expensive observation rendering (agent only uses agent_pos/dir)
    _dummy_obs = env.unwrapped.gen_obs()
    env.unwrapped.gen_obs = lambda: _dummy_obs

    _w_mapper = StateMapper(env)
    _w_dyn = DynamicsWrapper(env, seed=seed)
    _w_M_star = np.frombuffer(M_star_flat).reshape(n_states, n_states).copy()
    _w_env_seed = seed


def _run_single(args):
    """Worker: run 1 condition x 1 run x 3 phases."""
    (cond_name, run_idx, seed, n_conditions, cond_index,
     goal_positions, phase_episodes, max_ep_steps, config_dict) = args

    state_mapper = _w_mapper
    dyn = _w_dyn
    M_star = _w_M_star
    env_seed = _w_env_seed
    n_states = state_mapper.n_states

    agent_seed = seed + 10000 + run_idx * n_conditions + cond_index

    # Build config for PRISM (baselines ignore it)
    config = None
    if config_dict is not None:
        config = PRISMConfig(meta_sr=MetaSRConfig(**config_dict))

    agent = make_agent(cond_name, n_states, state_mapper, agent_seed,
                       config=config)

    goals = [tuple(g) for g in goal_positions]
    run_results = []

    for phase_idx, phase_num in enumerate([1, 2, 3]):
        dyn.apply_perturbation("reward_shift",
                               new_goal_pos=goals[phase_idx])

        for _ in range(phase_episodes):
            train_one_episode(dyn, agent, state_mapper,
                              env_seed, max_ep_steps)

        metrics = measure_calibration(agent, M_star)
        metrics["condition"] = cond_name
        metrics["run"] = run_idx
        metrics["phase"] = phase_num
        run_results.append(metrics)

    return (cond_name, run_idx, run_results)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def _save_results(all_results, output_dir, state_mapper):
    """Save CSV, u_snapshots, and reliability data."""
    # --- Main CSV ---
    csv_path = output_dir / "calibration_results.csv"
    fieldnames = [
        "condition", "run", "phase", "ece", "mi", "hl_stat", "hl_pvalue",
        "mean_confidence", "mean_error", "mean_uncertainty", "n_states_visited",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cond_name, runs_list in all_results.items():
            for run_phases in runs_list:
                for metrics in run_phases:
                    row = {k: metrics[k] for k in fieldnames}
                    for k in ["ece", "mi", "hl_stat", "hl_pvalue",
                              "mean_confidence", "mean_error",
                              "mean_uncertainty"]:
                        row[k] = f"{row[k]:.6f}"
                    writer.writerow(row)

    print(f"Results saved to {csv_path}")

    # --- U snapshots (run 0 only) ---
    u_dir = output_dir / "u_snapshots"
    u_dir.mkdir(exist_ok=True)

    for cond_name, runs_list in all_results.items():
        if len(runs_list) > 0:
            run_0_phases = runs_list[0]
            for metrics in run_0_phases:
                phase = metrics["phase"]
                U = metrics["_uncertainties"]
                U_grid = state_mapper.to_grid(U)
                np.save(u_dir / f"{cond_name}_U_phase{phase}.npy", U_grid)

    print(f"U snapshots saved to {u_dir}")

    # --- Reliability data (aggregated, phase 3 only) ---
    rel_dir = output_dir / "reliability"
    rel_dir.mkdir(exist_ok=True)

    for cond_name, runs_list in all_results.items():
        phase3_rels = []
        for run_phases in runs_list:
            for metrics in run_phases:
                if metrics["phase"] == 3:
                    phase3_rels.append(metrics["_reliability"])

        if phase3_rels:
            agg = aggregate_reliability(phase3_rels)
            rel_path = rel_dir / f"{cond_name}_phase3.json"
            with open(rel_path, "w") as f:
                json.dump(agg, f, indent=2)

    print(f"Reliability data saved to {rel_dir}")


def _print_summary(all_results):
    """Print summary table to console."""
    print("\n" + "=" * 80)
    print("SUMMARY — Experiment A: Calibration")
    print("=" * 80)
    print(f"{'Condition':<15s} {'ECE':>8s} {'MI':>8s} "
          f"{'HL pass%':>9s} {'MeanConf':>9s}")
    print("-" * 55)

    for cond_name, runs_list in all_results.items():
        phase3 = [m for run in runs_list for m in run if m["phase"] == 3]
        eces = [m["ece"] for m in phase3]
        mis = [m["mi"] for m in phase3]
        hl_pass = sum(1 for m in phase3 if m["hl_pvalue"] > 0.05) / len(phase3)
        mean_conf = np.mean([m["mean_confidence"] for m in phase3])

        print(f"{cond_name:<15s} {np.median(eces):>7.4f}  {np.median(mis):>7.4f}  "
              f"{hl_pass:>8.0%}  {mean_conf:>8.4f}")


def _run_statistics(all_results, output_dir):
    """Run statistical comparisons and save."""
    ece_dict = {}
    mi_dict = {}
    for cond_name, runs_list in all_results.items():
        phase3 = [m for run in runs_list for m in run if m["phase"] == 3]
        ece_dict[cond_name] = np.array([m["ece"] for m in phase3])
        mi_dict[cond_name] = np.array([m["mi"] for m in phase3])

    if "PRISM" not in ece_dict:
        return

    ece_comps = compare_conditions(ece_dict, reference="PRISM",
                                   alternative="less")
    mi_comps = compare_conditions(mi_dict, reference="PRISM",
                                  alternative="greater")

    print("\n" + "-" * 80)
    print("Statistical Tests — Phase 3")
    print("-" * 80)

    print("\nECE: PRISM < Baseline (one-sided)")
    for r in ece_comps:
        sig = "*" if r["p_corrected"] < 0.05 else ""
        print(f"  vs {r['condition']:<15s}  p_corr={r['p_corrected']:.4f}  {sig}")

    print("\nMI: PRISM > Baseline (one-sided)")
    for r in mi_comps:
        sig = "*" if r["p_corrected"] < 0.05 else ""
        print(f"  vs {r['condition']:<15s}  p_corr={r['p_corrected']:.4f}  {sig}")

    stats = {"ece_comparisons": ece_comps, "mi_comparisons": mi_comps}
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_exp_a(n_runs=100, seed=42, output_dir="results/exp_a",
              conditions=None, n_workers=None, note="",
              phase_episodes=200, max_ep_steps=500,
              config_override=None, resume_dir=None):
    """Run Experiment A: calibration comparison.

    Args:
        n_runs: Runs per condition.
        seed: Base random seed.
        output_dir: Root directory for this experiment's runs.
        conditions: List of condition names (default: all 5).
        n_workers: Number of parallel workers (default: auto).
        note: Free-text annotation.
        phase_episodes: Episodes per phase (default 200).
        max_ep_steps: Max steps per episode (default 500).
        config_override: Dict of MetaSR params to override for PRISM
                         (e.g. {"decay": 0.95, "beta": 5.0}).
        resume_dir: Path to a previous run directory to resume from.

    Returns:
        dict mapping condition -> list of run results.
    """
    if conditions is None:
        conditions = CONDITIONS

    if n_workers is None:
        n_workers = max(os.cpu_count() - 1, 1)

    # Build config dict for PRISM (None = defaults)
    config_dict = None
    if config_override:
        defaults = {
            "U_prior": 0.8, "decay": 0.85, "beta": 10.0, "theta_C": 0.3,
        }
        defaults.update(config_override)
        config_dict = defaults

    # --- Determine run directory (new or resumed) ---
    if resume_dir:
        run_dir = Path(resume_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        print(f"Resuming run: {run_dir}")
    else:
        run_timestamp = datetime.now()
        run_name = f"run_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run directory: {run_dir}")

    csv_path = run_dir / "calibration_results.csv"
    u_dir = run_dir / "u_snapshots"

    # --- Setup: env, mapper, M* ---
    print("Setting up environment...")
    env = gym.make("MiniGrid-FourRooms-v0")
    env.unwrapped.max_steps = max_ep_steps + 100
    env.reset(seed=seed)

    state_mapper = StateMapper(env)
    n_states = state_mapper.n_states
    print(f"  FourRooms: {n_states} accessible states")

    # Compute true SR matrix M* (constant: reward_shift doesn't change T)
    print("Computing true SR matrix M*...")
    dyn_wrapper = DynamicsWrapper(env)
    T = dyn_wrapper.get_true_transition_matrix(state_mapper)
    gamma = PRISMConfig().sr.gamma
    M_star = np.linalg.inv(np.eye(n_states) - gamma * T)
    print(f"  M* shape: {M_star.shape}, gamma={gamma}")

    # Flatten M_star for sharing via initializer
    M_star_flat = M_star.tobytes()

    # --- Pre-generate goal positions (3 per run, in different rooms) ---
    print("Pre-generating goal positions...")
    goal_positions_list = []
    for run_idx in range(n_runs):
        rng = np.random.default_rng(seed + run_idx)
        positions = generate_goal_positions(state_mapper, rng)
        goal_positions_list.append([list(p) for p in positions])

    # --- Build fine-grained task args ---
    n_conditions = len(conditions)
    n_tasks_total = n_conditions * n_runs
    total_episodes = n_tasks_total * phase_episodes * 3

    worker_args = []
    for cond_index, cond_name in enumerate(conditions):
        for run_idx in range(n_runs):
            worker_args.append((
                cond_name, run_idx, seed,
                n_conditions, cond_index,
                goal_positions_list[run_idx],
                phase_episodes, max_ep_steps,
                config_dict,
            ))

    # --- Filter out completed tasks on resume ---
    n_skipped = 0
    if resume_dir:
        completed_tasks = _load_completed_tasks(csv_path)
        before = len(worker_args)
        worker_args = [
            a for a in worker_args
            if (a[0], a[1]) not in completed_tasks
        ]
        n_skipped = before - len(worker_args)
        if n_skipped:
            print(f"  Resume: {n_skipped} tasks already done, "
                  f"{len(worker_args)} remaining")

    n_tasks = len(worker_args)

    if n_tasks == 0:
        print("All tasks already completed — skipping to post-processing.")
    else:
        print(f"\n{n_tasks} tasks "
              f"({n_conditions} conditions x {n_runs} runs"
              f"{f', {n_skipped} skipped' if n_skipped else ''}, "
              f"{n_workers} workers)")
        if config_dict:
            print(f"PRISM config: {config_dict}")
        print()

    # --- Write run_info.json at start ---
    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": "exp_a",
        "seed": seed,
        "n_runs": n_runs,
        "phase_episodes": phase_episodes,
        "max_ep_steps": max_ep_steps,
        "n_workers": n_workers,
        "conditions": list(conditions),
        "n_conditions": len(conditions),
        "total_episodes": total_episodes,
        "note": note,
        "status": "running",
    }
    if config_dict:
        run_info["prism_config"] = config_dict
    if resume_dir:
        run_info["resumed_from"] = str(resume_dir)
        run_info["n_skipped"] = n_skipped

    run_info_path = run_dir / "run_info.json"
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)

    # --- Progress tracking ---
    progress_path = run_dir / "_progress.json"
    t_start = time.time()

    def _write_progress(completed):
        elapsed = time.time() - t_start
        remaining = n_tasks - completed
        eta = (elapsed / max(completed, 1)) * remaining
        prog = {
            "completed": completed + n_skipped,
            "total": n_tasks_total,
            "pct": round(100 * (completed + n_skipped) / n_tasks_total, 1),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
        }
        with open(progress_path, "w") as fp:
            json.dump(prog, fp)

    # --- Incremental save callback ---
    def _on_task_done(cond_name, run_idx, run_phases):
        _append_csv_rows(csv_path, run_phases)
        _save_u_snapshot_if_run0(cond_name, run_idx, run_phases,
                                 u_dir, state_mapper)

    # --- Execute ---
    raw_results = []

    if n_tasks > 0:
        _write_progress(0)

        if n_workers == 1:
            _init_worker(seed, M_star_flat, n_states, max_ep_steps)
            for i, args in enumerate(tqdm(worker_args, desc="Exp A")):
                result = _run_single(args)
                raw_results.append(result)
                _on_task_done(*result)
                _write_progress(i + 1)
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(seed, M_star_flat, n_states, max_ep_steps),
            ) as executor:
                futures = {executor.submit(_run_single, args): idx
                           for idx, args in enumerate(worker_args)}
                for i, fut in enumerate(tqdm(
                        as_completed(futures), total=n_tasks, desc="Exp A")):
                    result = fut.result()
                    raw_results.append(result)
                    _on_task_done(*result)
                    _write_progress(i + 1)

        elapsed = time.time() - t_start
        print(f"\nDone in {elapsed:.1f}s "
              f"({elapsed/n_tasks:.2f}s per task)")
    else:
        elapsed = 0.0

    # --- Post-processing ---

    # Sort CSV for clean output
    _rewrite_csv_sorted(csv_path, conditions)

    # Rebuild all_results from the complete CSV (covers all sessions)
    all_results = _load_results_from_csv(csv_path, conditions, n_runs)

    # Print summary and statistics from full CSV data
    _print_summary(all_results)
    _run_statistics(all_results, run_dir)

    # Reliability (current session only — previous session data is lost)
    rel_dir = run_dir / "reliability"
    if raw_results:
        session_results = defaultdict(list)
        for cond_name, run_idx, run_phases in raw_results:
            session_results[cond_name].append(run_phases)

        rel_dir.mkdir(exist_ok=True)
        for cond_name in conditions:
            if cond_name not in session_results:
                continue
            phase3_rels = []
            for run_phases in session_results[cond_name]:
                for m in run_phases:
                    if m["phase"] == 3:
                        phase3_rels.append(m["_reliability"])
            if phase3_rels:
                agg = aggregate_reliability(phase3_rels)
                with open(rel_dir / f"{cond_name}_phase3.json", "w") as f:
                    json.dump(agg, f, indent=2)

        if resume_dir and n_skipped > 0:
            print(f"Warning: reliability data covers this session's "
                  f"{len(raw_results)} tasks only (previous session's "
                  f"in-memory data was lost).")
        print(f"Reliability data saved to {rel_dir}")

    # --- Finalize run_info ---
    run_info["status"] = "completed"
    run_info["elapsed_seconds"] = round(elapsed, 1)
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nRun info saved to {run_info_path}")

    # Clean up progress file
    if progress_path.exists():
        progress_path.unlink()

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment A: Calibration (P1+P2)")
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/exp_a")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to run")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--note", type=str, default="",
                        help="Free-text note for this run")
    parser.add_argument("--phase_episodes", type=int, default=200,
                        help="Episodes per phase (default 200)")
    parser.add_argument("--max_ep_steps", type=int, default=500,
                        help="Max steps per episode (default 500)")
    # MetaSR override parameters (from sweep A.2)
    parser.add_argument("--decay", type=float, default=None,
                        help="MetaSR decay override (sweep: 0.95)")
    parser.add_argument("--beta", type=float, default=None,
                        help="MetaSR beta override (sweep: 5.0)")
    parser.add_argument("--U_prior", type=float, default=None,
                        help="MetaSR U_prior override")
    parser.add_argument("--theta_C", type=float, default=None,
                        help="MetaSR theta_C override")
    parser.add_argument("--resume", type=str, default=None,
                        metavar="RUN_DIR",
                        help="Resume a previous run (path to run directory)")
    args = parser.parse_args()

    # Build config override dict from CLI args
    config_override = {}
    for param in ["decay", "beta", "U_prior", "theta_C"]:
        val = getattr(args, param)
        if val is not None:
            config_override[param] = val

    run_exp_a(
        n_runs=args.n_runs,
        seed=args.seed,
        output_dir=args.output_dir,
        conditions=args.conditions,
        n_workers=args.workers,
        note=args.note,
        phase_episodes=args.phase_episodes,
        max_ep_steps=args.max_ep_steps,
        config_override=config_override or None,
        resume_dir=args.resume,
    )


if __name__ == "__main__":
    main()
