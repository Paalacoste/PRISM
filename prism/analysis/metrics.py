"""Statistical analysis: bootstrap CI, Mann-Whitney U, Holm-Bonferroni.

References:
    master.md section 6 (statistical protocol)
"""

import numpy as np
from scipy import stats
from itertools import combinations


def bootstrap_ci(data, n_resamples=10000, confidence=0.95, statistic=np.mean,
                 seed=42):
    """Compute bootstrap confidence interval using percentile method.

    Args:
        data: 1D array of observations.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        statistic: Function to compute on each resample (default: mean).
        seed: Random seed.

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    point = float(statistic(data))

    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = statistic(sample)

    alpha = 1 - confidence
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return (point, ci_low, ci_high)


def mann_whitney_test(x, y, alternative="two-sided"):
    """Mann-Whitney U test with rank-biserial effect size.

    Args:
        x, y: Two independent samples.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        dict with keys: U, p_value, effect_size (rank-biserial r).
    """
    x, y = np.asarray(x), np.asarray(y)
    result = stats.mannwhitneyu(x, y, alternative=alternative)
    U = result.statistic
    n1, n2 = len(x), len(y)
    # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
    r = 1 - 2 * U / (n1 * n2)
    return {"U": float(U), "p_value": float(result.pvalue),
            "effect_size": float(r)}


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: list of p-values.

    Returns:
        list of corrected p-values (same order as input).
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n

    cumulative_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        # Enforce monotonicity
        cumulative_max = max(cumulative_max, adjusted)
        corrected[orig_idx] = min(cumulative_max, 1.0)

    return corrected


def compare_conditions(results_dict, reference="PRISM", alternative="two-sided"):
    """Pairwise Mann-Whitney U tests of reference vs all other conditions.

    Args:
        results_dict: dict mapping condition_name -> 1D array of metric values.
        reference: Name of the reference condition (default: PRISM).
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        list of dicts, each with:
            condition, U, p_value, p_corrected, effect_size, ref_mean, cond_mean
    """
    ref_data = np.asarray(results_dict[reference])
    others = [k for k in results_dict if k != reference]

    raw_results = []
    for cond in others:
        cond_data = np.asarray(results_dict[cond])
        test = mann_whitney_test(ref_data, cond_data, alternative=alternative)
        raw_results.append({
            "condition": cond,
            **test,
            "ref_mean": float(np.mean(ref_data)),
            "cond_mean": float(np.mean(cond_data)),
        })

    # Holm-Bonferroni correction
    p_values = [r["p_value"] for r in raw_results]
    corrected = holm_bonferroni(p_values)

    for r, p_corr in zip(raw_results, corrected):
        r["p_corrected"] = p_corr

    return sorted(raw_results, key=lambda r: r["p_corrected"])


def compare_all_pairs(results_dict):
    """All pairwise Mann-Whitney U tests with Holm-Bonferroni correction.

    Args:
        results_dict: dict mapping condition_name -> 1D array of metric values.

    Returns:
        list of dicts with: cond_a, cond_b, U, p_value, p_corrected, effect_size
    """
    conditions = sorted(results_dict.keys())
    raw_results = []

    for a, b in combinations(conditions, 2):
        test = mann_whitney_test(results_dict[a], results_dict[b])
        raw_results.append({
            "cond_a": a,
            "cond_b": b,
            **test,
            "mean_a": float(np.mean(results_dict[a])),
            "mean_b": float(np.mean(results_dict[b])),
        })

    p_values = [r["p_value"] for r in raw_results]
    corrected = holm_bonferroni(p_values)
    for r, p_corr in zip(raw_results, corrected):
        r["p_corrected"] = p_corr

    return sorted(raw_results, key=lambda r: r["p_corrected"])
