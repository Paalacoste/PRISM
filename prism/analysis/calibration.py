"""Psychophysics calibration metrics: ECE, reliability diagrams, MI.

ECE (Expected Calibration Error):
    ECE = sum_{b=1}^{n_bins} (|B_b|/N) * |accuracy_b - confidence_b|

Accuracy definition for SR:
    accuracy(s) = 1 if ||M(s,:) - M*(s,:)||_2 < tau, else 0
    tau = 50th percentile of all errors (ensures ~50% baseline accuracy)

Metacognitive Index (MI):
    MI = Spearman correlation between U(s) and ||M(s,:) - M*(s,:)||_2

References:
    master.md section 6.1
"""

import numpy as np
from scipy import stats


def sr_errors(M, M_star):
    """Compute per-state SR errors: ||M(s,:) - M*(s,:)||_2.

    Args:
        M: Learned SR matrix of shape (n_states, n_states).
        M_star: Ground-truth SR matrix of shape (n_states, n_states).

    Returns:
        Array of shape (n_states,) with L2 error per state.
    """
    return np.linalg.norm(M - M_star, axis=1)


def sr_accuracies(M, M_star, percentile=50):
    """Compute binary accuracy per state.

    accuracy(s) = 1 if error(s) < tau, else 0
    tau = given percentile of all errors.

    Args:
        M: Learned SR matrix.
        M_star: Ground-truth SR matrix.
        percentile: Percentile for accuracy threshold (default 50).

    Returns:
        Binary array of shape (n_states,).
    """
    errors = sr_errors(M, M_star)
    tau = np.percentile(errors, percentile)
    return (errors < tau).astype(float)


def expected_calibration_error(confidences, accuracies, n_bins=10):
    """Compute Expected Calibration Error (ECE).

    ECE = sum_{b} (|B_b|/N) * |accuracy_b - confidence_b|

    Args:
        confidences: Array of confidence values in [0, 1].
        accuracies: Binary array (0 or 1) of same length.
        n_bins: Number of bins (default 10).

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated).
    """
    confidences = np.asarray(confidences)
    accuracies = np.asarray(accuracies)
    N = len(confidences)

    if N == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        # Include upper bound in last bin
        if i == n_bins - 1:
            mask = mask | (confidences == bin_boundaries[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_confidence = confidences[mask].mean()
            avg_accuracy = accuracies[mask].mean()
            ece += (n_in_bin / N) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def reliability_diagram_data(confidences, accuracies, n_bins=10):
    """Compute data for a reliability diagram.

    Args:
        confidences: Array of confidence values in [0, 1].
        accuracies: Binary array (0 or 1).
        n_bins: Number of bins.

    Returns:
        Dict with keys:
            bin_confidences: Mean confidence per bin.
            bin_accuracies: Mean accuracy per bin.
            bin_counts: Number of samples per bin.
            bin_centers: Center of each bin.
    """
    confidences = np.asarray(confidences)
    accuracies = np.asarray(accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    bin_centers = []

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = mask | (confidences == bin_boundaries[i + 1])

        n_in_bin = mask.sum()
        bin_counts.append(int(n_in_bin))
        bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)

        if n_in_bin > 0:
            bin_confidences.append(float(confidences[mask].mean()))
            bin_accuracies.append(float(accuracies[mask].mean()))
        else:
            bin_confidences.append(float(bin_centers[-1]))
            bin_accuracies.append(0.0)

    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "bin_centers": bin_centers,
    }


def plot_reliability_diagram(confidences, accuracies, n_bins=10, ax=None):
    """Plot a reliability diagram.

    Args:
        confidences: Array of confidence values.
        accuracies: Binary array.
        n_bins: Number of bins.
        ax: Optional matplotlib axes.

    Returns:
        fig: The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    data = reliability_diagram_data(confidences, accuracies, n_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Bars for model calibration
    width = 1.0 / n_bins * 0.8
    non_empty = [i for i, c in enumerate(data["bin_counts"]) if c > 0]
    ax.bar(
        [data["bin_centers"][i] for i in non_empty],
        [data["bin_accuracies"][i] for i in non_empty],
        width=width,
        alpha=0.7,
        color="steelblue",
        label="PRISM",
    )

    ece = expected_calibration_error(confidences, accuracies, n_bins)
    ax.set_xlabel("Mean Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.3f})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    fig.tight_layout()
    return fig


def metacognitive_index(U, M, M_star):
    """Compute Metacognitive Index: Spearman correlation between U and true error.

    MI = rho_Spearman(U(s), ||M(s,:) - M*(s,:)||_2)

    Args:
        U: Uncertainty array of shape (n_states,).
        M: Learned SR matrix.
        M_star: Ground-truth SR matrix.

    Returns:
        Tuple of (rho, p_value). rho in [-1, 1], higher is better.
    """
    errors = sr_errors(M, M_star)

    # Handle edge case: constant values
    if np.std(U) == 0 or np.std(errors) == 0:
        return (0.0, 1.0)

    rho, p_value = stats.spearmanr(U, errors)
    return (float(rho), float(p_value))
