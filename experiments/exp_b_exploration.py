"""Experiment B: Exploration efficiency (tests Proposition P3).

Protocol:
    - FourRooms 19x19, 4 hidden goals (one per room)
    - 8 exploration strategies compared
    - 100 runs per condition, same goal placements across conditions
    - Metrics: steps to all goals, coverage, redundancy, efficiency ratio
    - Statistical tests: Mann-Whitney U with Holm-Bonferroni correction

Usage:
    python -m experiments.exp_b_exploration [--n_runs 100] [--max_steps 2000] [--seed 42]
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import minigrid  # noqa: F401 — must import before gym.make
import gymnasium as gym

from prism.agent.sr_layer import SRLayer
from prism.agent.meta_sr import MetaSR
from prism.agent.controller import PRISMController
from prism.config import PRISMConfig
from prism.env.state_mapper import StateMapper
from prism.env.dynamics_wrapper import DynamicsWrapper
from prism.env.exploration_task import place_goals
from prism.baselines import (
    RandomAgent,
    SREpsilonGreedy,
    SREpsilonDecay,
    SRCountBonus,
    SRNormBonus,
    SRPosterior,
    SROracle,
)
from prism.analysis.metrics import bootstrap_ci, compare_conditions


# MiniGrid movement actions
MOVEMENT_ACTIONS = [0, 1, 2]


# ---------------------------------------------------------------------------
# PRISM adapter for common experiment loop
# ---------------------------------------------------------------------------
class PRISMExpBAgent:
    """Adapts PRISM components (SR + MetaSR + Controller) for Exp B loop."""

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

    def exploration_bonus(self, s):
        return self.meta_sr.uncertainty(s)

    def exploration_value(self, s):
        return self.controller.exploration_value(s)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------
def run_single_episode(env, agent, state_mapper, goal_positions, max_steps,
                       env_seed):
    """Run one exploration episode. Returns metrics dict.

    Args:
        env: MiniGrid gymnasium env (unwrapped or base).
        agent: Any agent with select_action(s, actions, dir) and update(s, s', r).
        state_mapper: StateMapper instance.
        goal_positions: list of (x, y) goal positions.
        max_steps: Maximum steps before forced stop.
        env_seed: Fixed seed for env.reset() (preserves grid structure).

    Returns:
        dict with: steps, goals_found, all_found, coverage, redundancy,
                   discovery_times (list of ints, max_steps if not found).
    """
    # Always use the same env seed to preserve grid structure (wall positions).
    # FourRooms randomizes passage locations per seed, so changing the seed
    # would invalidate the StateMapper.
    obs, info = env.reset(seed=env_seed)
    env.unwrapped.max_steps = max_steps + 100  # prevent MiniGrid truncation

    # Remove built-in goal to prevent MiniGrid termination
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                grid.set(x, y, None)

    goal_set = {tuple(p) for p in goal_positions}
    discovered = set()
    discovery_times = {}
    visited_states = set()
    step = 0

    s = state_mapper.get_index(tuple(env.unwrapped.agent_pos))

    for step in range(1, max_steps + 1):
        visited_states.add(s)

        agent_dir = env.unwrapped.agent_dir
        action = agent.select_action(s, MOVEMENT_ACTIONS, agent_dir)
        obs, _reward, terminated, truncated, info = env.step(action)

        s_next = state_mapper.get_index(tuple(env.unwrapped.agent_pos))
        pos = tuple(env.unwrapped.agent_pos)

        # Goal discovery check
        reward = 0.0
        if pos in goal_set and pos not in discovered:
            discovered.add(pos)
            discovery_times[pos] = step
            reward = 1.0

        agent.update(s, s_next, reward)
        s = s_next

        # All goals found
        if len(discovered) == len(goal_set):
            break

    visited_states.add(s)

    # Build ordered discovery times
    sorted_goals = sorted(goal_positions)
    disc_list = [discovery_times.get(tuple(g), max_steps + 1)
                 for g in sorted_goals]

    return {
        "steps": step,
        "goals_found": len(discovered),
        "all_found": len(discovered) == len(goal_set),
        "coverage": len(visited_states) / state_mapper.n_states,
        "redundancy": step / max(len(visited_states), 1),
        "discovery_times": disc_list,
    }


# ---------------------------------------------------------------------------
# Condition factory
# ---------------------------------------------------------------------------
def make_agent(condition, n_states, state_mapper, M_star, seed):
    """Create an agent for the given condition name."""
    common = dict(n_states=n_states, state_mapper=state_mapper, seed=seed)

    if condition == "PRISM":
        return PRISMExpBAgent(n_states, state_mapper, seed=seed)
    elif condition == "SR-Oracle":
        return SROracle(M_star=M_star, **common)
    elif condition == "SR-e-greedy":
        return SREpsilonGreedy(epsilon=0.1, **common)
    elif condition == "SR-e-decay":
        return SREpsilonDecay(epsilon_start=0.5, epsilon_min=0.01,
                              decay_rate=0.999, **common)
    elif condition == "SR-Count-Bonus":
        return SRCountBonus(epsilon=0.1, **common)
    elif condition == "SR-Norm-Bonus":
        return SRNormBonus(epsilon=0.1, **common)
    elif condition == "SR-Posterior":
        return SRPosterior(epsilon=0.1, **common)
    elif condition == "Random":
        return RandomAgent(**common)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
CONDITIONS = [
    "PRISM",
    "SR-Oracle",
    "SR-e-greedy",
    "SR-e-decay",
    "SR-Count-Bonus",
    "SR-Norm-Bonus",
    "SR-Posterior",
    "Random",
]


def run_exp_b(n_runs=100, max_steps=2000, seed=42,
              output_dir="results/exp_b", conditions=None):
    """Run Experiment B: exploration efficiency comparison.

    Args:
        n_runs: Number of runs per condition.
        max_steps: Max steps per episode.
        seed: Base random seed.
        output_dir: Directory for results.
        conditions: List of condition names (default: all 8).

    Returns:
        dict mapping condition -> list of metric dicts.
    """
    if conditions is None:
        conditions = CONDITIONS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup: create env, mapper, true SR ---
    print("Setting up environment...")
    env = gym.make("MiniGrid-FourRooms-v0")
    env.unwrapped.max_steps = max_steps + 100  # prevent MiniGrid truncation
    obs, _ = env.reset(seed=seed)

    state_mapper = StateMapper(env)
    n_states = state_mapper.n_states
    print(f"  FourRooms: {n_states} accessible states")

    # Compute true SR matrix M* for Oracle baseline
    print("Computing true SR matrix M*...")
    dyn_wrapper = DynamicsWrapper(env)
    T = dyn_wrapper.get_true_transition_matrix(state_mapper)
    gamma = PRISMConfig().sr.gamma
    M_star = np.linalg.inv(np.eye(n_states) - gamma * T)
    print(f"  M* shape: {M_star.shape}, gamma={gamma}")

    # --- Pre-generate goal placements (same across conditions) ---
    print("Pre-generating goal placements...")
    goal_placements = []
    for run_idx in range(n_runs):
        goal_rng = np.random.default_rng(seed + run_idx)
        goals = place_goals(state_mapper, n_goals=4, rng=goal_rng)
        goal_placements.append(goals)

    # --- Run all conditions ---
    all_results = {}
    total_runs = len(conditions) * n_runs
    print(f"\nRunning {total_runs} episodes "
          f"({len(conditions)} conditions x {n_runs} runs)...\n")

    t_start = time.time()

    for cond_name in conditions:
        cond_results = []
        desc = f"{cond_name:20s}"

        for run_idx in tqdm(range(n_runs), desc=desc, leave=True):
            # Fixed env seed: grid structure must be constant across runs
            # (FourRooms randomizes passage positions per seed).
            # Variation comes from goal placement and agent seed.
            env_seed = seed
            agent_seed = seed + 10000 + run_idx * len(conditions) + \
                conditions.index(cond_name)

            agent = make_agent(
                cond_name, n_states, state_mapper, M_star, agent_seed,
            )

            metrics = run_single_episode(
                env, agent, state_mapper,
                goal_placements[run_idx], max_steps, env_seed,
            )
            metrics["condition"] = cond_name
            metrics["run"] = run_idx
            cond_results.append(metrics)

        all_results[cond_name] = cond_results

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/total_runs:.3f}s per run)")

    # --- Save results ---
    _save_results(all_results, output_dir, n_runs)

    # --- Print summary ---
    _print_summary(all_results, max_steps)

    # --- Statistical tests ---
    _run_statistics(all_results, output_dir)

    return all_results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------
def _save_results(all_results, output_dir, n_runs):
    """Save results as CSV."""
    csv_path = output_dir / "exploration_results.csv"
    fieldnames = [
        "condition", "run", "steps", "goals_found", "all_found",
        "coverage", "redundancy",
        "discovery_1", "discovery_2", "discovery_3", "discovery_4",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cond_name, runs in all_results.items():
            for m in runs:
                row = {
                    "condition": m["condition"],
                    "run": m["run"],
                    "steps": m["steps"],
                    "goals_found": m["goals_found"],
                    "all_found": m["all_found"],
                    "coverage": f"{m['coverage']:.4f}",
                    "redundancy": f"{m['redundancy']:.4f}",
                }
                for i, t in enumerate(m["discovery_times"]):
                    row[f"discovery_{i+1}"] = t
                writer.writerow(row)

    print(f"\nResults saved to {csv_path}")

    # --- Summary CSV (one row per condition) ---
    summary_path = output_dir / "summary.csv"
    summary_fields = [
        "condition", "n_runs", "steps_mean", "steps_median", "steps_ci_lo",
        "steps_ci_hi", "goals_mean", "all_found_pct", "coverage_mean",
        "redundancy_mean",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for cond_name, runs in all_results.items():
            steps = [m["steps"] for m in runs]
            mean_s, ci_lo, ci_hi = bootstrap_ci(steps)
            writer.writerow({
                "condition": cond_name,
                "n_runs": len(runs),
                "steps_mean": f"{mean_s:.1f}",
                "steps_median": f"{np.median(steps):.1f}",
                "steps_ci_lo": f"{ci_lo:.1f}",
                "steps_ci_hi": f"{ci_hi:.1f}",
                "goals_mean": f"{np.mean([m['goals_found'] for m in runs]):.2f}",
                "all_found_pct": f"{np.mean([m['all_found'] for m in runs]):.4f}",
                "coverage_mean": f"{np.mean([m['coverage'] for m in runs]):.4f}",
                "redundancy_mean": f"{np.mean([m['redundancy'] for m in runs]):.4f}",
            })
    print(f"Summary saved to {summary_path}")


def _print_summary(all_results, max_steps):
    """Print summary table to console."""
    print("\n" + "=" * 80)
    print("SUMMARY — Experiment B: Exploration Efficiency")
    print("=" * 80)
    print(f"{'Condition':<20s} {'Steps':>8s} {'Goals':>6s} "
          f"{'Coverage':>9s} {'Redundancy':>11s}")
    print("-" * 60)

    summary = {}
    for cond_name, runs in all_results.items():
        steps = [m["steps"] for m in runs]
        goals = [m["goals_found"] for m in runs]
        cov = [m["coverage"] for m in runs]
        red = [m["redundancy"] for m in runs]

        mean_steps, ci_lo, ci_hi = bootstrap_ci(steps)
        mean_goals = np.mean(goals)
        mean_cov = np.mean(cov)
        mean_red = np.mean(red)

        summary[cond_name] = {
            "mean_steps": mean_steps,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "mean_goals": mean_goals,
            "mean_coverage": mean_cov,
            "mean_redundancy": mean_red,
        }

        print(f"{cond_name:<20s} {mean_steps:>7.0f}  {mean_goals:>5.1f}  "
              f"{mean_cov:>8.1%}  {mean_red:>10.2f}")

    # Efficiency ratio
    if "Random" in summary and "SR-Oracle" in summary and "PRISM" in summary:
        s_rand = summary["Random"]["mean_steps"]
        s_oracle = summary["SR-Oracle"]["mean_steps"]
        s_prism = summary["PRISM"]["mean_steps"]
        if s_rand != s_oracle:
            eff = (s_rand - s_prism) / (s_rand - s_oracle)
            print(f"\nEfficiency ratio (PRISM): {eff:.2f}")
            print(f"  (1.0 = matches Oracle, 0.0 = matches Random)")


def _run_statistics(all_results, output_dir):
    """Run statistical comparisons and save."""
    # Extract steps arrays per condition
    steps_dict = {}
    for cond_name, runs in all_results.items():
        steps_dict[cond_name] = np.array([m["steps"] for m in runs])

    if "PRISM" not in steps_dict:
        return

    # PRISM vs all — one-sided (PRISM should have fewer steps)
    comparisons = compare_conditions(
        steps_dict, reference="PRISM", alternative="less",
    )

    print("\n" + "-" * 80)
    print("Statistical Tests: PRISM vs each condition (Mann-Whitney U, one-sided)")
    print("-" * 80)
    print(f"{'Condition':<20s} {'PRISM':>8s} {'Other':>8s} "
          f"{'p-value':>10s} {'p-corr':>10s} {'Effect':>8s} {'Sig':>5s}")
    print("-" * 80)

    for r in comparisons:
        sig = "*" if r["p_corrected"] < 0.05 else ""
        print(f"{r['condition']:<20s} {r['ref_mean']:>7.0f}  {r['cond_mean']:>7.0f}  "
              f"{r['p_value']:>9.4f}  {r['p_corrected']:>9.4f}  "
              f"{r['effect_size']:>+7.3f}  {sig:>4s}")

    # Save statistics as JSON
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment B: Exploration")
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/exp_b")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to run")
    args = parser.parse_args()

    run_exp_b(
        n_runs=args.n_runs,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
