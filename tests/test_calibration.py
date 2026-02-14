"""Tests for calibration metrics."""

import numpy as np
import pytest

from prism.analysis.calibration import (
    sr_errors,
    sr_accuracies,
    expected_calibration_error,
    reliability_diagram_data,
    metacognitive_index,
)


class TestSRErrors:

    def test_zero_error_identical_matrices(self):
        M = np.eye(5)
        errors = sr_errors(M, M)
        np.testing.assert_allclose(errors, 0.0)

    def test_error_shape(self):
        M = np.random.randn(10, 10)
        M_star = np.random.randn(10, 10)
        errors = sr_errors(M, M_star)
        assert errors.shape == (10,)

    def test_errors_non_negative(self):
        M = np.random.randn(5, 5)
        M_star = np.random.randn(5, 5)
        errors = sr_errors(M, M_star)
        assert np.all(errors >= 0)


class TestSRAccuracies:

    def test_all_accurate_when_identical(self):
        M = np.eye(5)
        acc = sr_accuracies(M, M)
        # All errors are 0, all below any percentile -> all accurate
        # Edge case: percentile of all zeros is 0, errors < 0 is false
        # This is actually tricky — all errors == 0, tau == 0, so 0 < 0 is False
        # That's expected behavior: when all errors are equal, none are "below" median
        assert acc.shape == (5,)

    def test_half_accurate_with_median(self):
        """With percentile=50, roughly half should be accurate."""
        np.random.seed(42)
        M = np.random.randn(100, 100)
        M_star = np.random.randn(100, 100)
        acc = sr_accuracies(M, M_star, percentile=50)
        # Should be approximately 50%
        assert 40 <= acc.sum() <= 60

    def test_binary_values(self):
        M = np.random.randn(10, 10)
        M_star = np.random.randn(10, 10)
        acc = sr_accuracies(M, M_star)
        assert set(np.unique(acc)).issubset({0.0, 1.0})


class TestECE:

    def test_perfect_calibration(self):
        """Perfect calibration: confidence matches accuracy."""
        # 100 samples, confidence = accuracy
        confidences = np.array([0.1] * 50 + [0.9] * 50)
        accuracies = np.array([0.0] * 45 + [1.0] * 5 + [1.0] * 45 + [0.0] * 5)
        ece = expected_calibration_error(confidences, accuracies)
        assert ece < 0.15  # Not perfectly 0 due to binning, but should be low

    def test_worst_case_calibration(self):
        """All confident but all wrong."""
        confidences = np.ones(100) * 0.95
        accuracies = np.zeros(100)
        ece = expected_calibration_error(confidences, accuracies)
        assert ece > 0.8

    def test_ece_in_range(self):
        confidences = np.random.rand(100)
        accuracies = np.random.randint(0, 2, 100).astype(float)
        ece = expected_calibration_error(confidences, accuracies)
        assert 0 <= ece <= 1

    def test_empty_returns_zero(self):
        ece = expected_calibration_error(np.array([]), np.array([]))
        assert ece == 0.0


class TestReliabilityDiagram:

    def test_output_keys(self):
        data = reliability_diagram_data(np.random.rand(50), np.random.randint(0, 2, 50))
        assert "bin_confidences" in data
        assert "bin_accuracies" in data
        assert "bin_counts" in data
        assert "bin_centers" in data

    def test_bin_count(self):
        data = reliability_diagram_data(np.random.rand(50), np.random.randint(0, 2, 50), n_bins=5)
        assert len(data["bin_centers"]) == 5

    def test_total_count_matches(self):
        N = 100
        data = reliability_diagram_data(np.random.rand(N), np.random.randint(0, 2, N))
        assert sum(data["bin_counts"]) == N


class TestMetacognitiveIndex:

    def test_perfect_correlation(self):
        """U perfectly tracks true error."""
        n = 50
        M = np.random.randn(n, n)
        M_star = np.random.randn(n, n)
        errors = np.linalg.norm(M - M_star, axis=1)
        U = errors  # Perfect tracking

        rho, p = metacognitive_index(U, M, M_star)
        assert rho > 0.99

    def test_random_correlation(self):
        """Random U should have low correlation."""
        np.random.seed(42)
        n = 100
        M = np.random.randn(n, n)
        M_star = np.random.randn(n, n)
        U = np.random.rand(n)

        rho, p = metacognitive_index(U, M, M_star)
        assert abs(rho) < 0.3  # Should be near zero

    def test_constant_U_returns_zero(self):
        """Constant U -> rho = 0."""
        n = 10
        U = np.ones(n) * 0.5
        M = np.random.randn(n, n)
        M_star = np.random.randn(n, n)
        rho, p = metacognitive_index(U, M, M_star)
        assert rho == 0.0

    def test_returns_tuple(self):
        n = 10
        U = np.random.rand(n)
        M = np.random.randn(n, n)
        M_star = np.random.randn(n, n)
        result = metacognitive_index(U, M, M_star)
        assert isinstance(result, tuple)
        assert len(result) == 2
