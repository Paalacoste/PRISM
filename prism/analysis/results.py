"""Run catalog utilities — list, load, and find experiment runs.

Each run is stored in results/{experiment}/run_YYYYMMDD_HHMMSS/ with a
run_info.json metadata file.
"""

import json
from pathlib import Path

import pandas as pd


def get_latest_run(experiment="exp_b", results_root="results"):
    """Return the Path of the most recent run for this experiment.

    Scans results/{experiment}/run_*/run_info.json and returns the
    directory with the latest timestamp (lexicographic sort on dir name).

    Args:
        experiment: Experiment name (e.g. "exp_b").
        results_root: Root results directory.

    Returns:
        Path to the run directory.

    Raises:
        FileNotFoundError: If no cataloged runs exist.
    """
    exp_dir = Path(results_root) / experiment
    run_dirs = sorted(exp_dir.glob("run_*"), reverse=True)

    for d in run_dirs:
        if (d / "run_info.json").exists():
            return d

    raise FileNotFoundError(
        f"No cataloged runs found in {exp_dir}. "
        f"Expected directories matching run_*/run_info.json"
    )


def list_runs(experiment="exp_b", results_root="results"):
    """List all cataloged runs with date, conditions, and note.

    Returns:
        list of dicts with keys: path, timestamp, conditions, n_runs, note.
    """
    exp_dir = Path(results_root) / experiment
    runs = []

    for d in sorted(exp_dir.glob("run_*")):
        info_path = d / "run_info.json"
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        runs.append({
            "path": d,
            "timestamp": info.get("timestamp", "?"),
            "conditions": info.get("conditions", []),
            "n_runs": info.get("n_runs", 0),
            "note": info.get("note", ""),
        })

    return runs


_CSV_DEFAULTS = {
    "exp_a": "calibration_results.csv",
    "exp_b": "exploration_results.csv",
}


def load_run(run_dir, csv_name=None):
    """Load a run's DataFrame and metadata.

    Args:
        run_dir: Path to the run directory (contains run_info.json + CSV).
        csv_name: CSV filename override. If None, auto-detected from
            run_info["experiment"] (exp_a -> calibration_results.csv,
            exp_b -> exploration_results.csv).

    Returns:
        (df, run_info) tuple.
    """
    run_dir = Path(run_dir)

    with open(run_dir / "run_info.json") as f:
        run_info = json.load(f)

    if csv_name is None:
        experiment = run_info.get("experiment", "exp_b")
        csv_name = _CSV_DEFAULTS.get(experiment, "exploration_results.csv")

    csv_path = run_dir / csv_name
    df = pd.read_csv(csv_path)

    if "coverage" in df.columns:
        df["coverage"] = df["coverage"].astype(float)
    if "redundancy" in df.columns:
        df["redundancy"] = df["redundancy"].astype(float)

    return df, run_info
