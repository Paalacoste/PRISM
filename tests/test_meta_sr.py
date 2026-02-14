"""Tests for MetaSR."""

import numpy as np
import pytest

from prism.agent.meta_sr import MetaSR


class TestMetaSRInit:

    def test_default_params(self):
        m = MetaSR(n_states=10)
        assert m.n_states == 10
        assert m.buffer_size == 20
        assert m.U_prior == 0.8
        assert m.decay == 0.85
        assert m.beta == 10.0
        assert m.theta_C == 0.3
        assert m.theta_change == 0.5

    def test_visit_counts_zero(self):
        m = MetaSR(n_states=5)
        np.testing.assert_array_equal(m.visit_counts, np.zeros(5))


class TestMetaSRUncertainty:

    def test_unvisited_returns_prior(self):
        m = MetaSR(n_states=5, U_prior=0.8)
        assert m.uncertainty(0) == 0.8
        assert m.uncertainty(4) == 0.8

    def test_cold_start_decay(self):
        """After a few visits (< K), U decays exponentially."""
        m = MetaSR(n_states=3, buffer_size=20, U_prior=0.8, decay=0.85)
        delta = np.ones(3)  # dummy delta

        # Visit state 0 five times
        for _ in range(5):
            m.observe(0, delta)

        expected = 0.8 * (0.85 ** 5)
        assert m.uncertainty(0) == pytest.approx(expected, rel=1e-6)

    def test_sufficient_data_uses_buffer_mean(self):
        """After K visits, U = mean of buffer."""
        K = 5
        m = MetaSR(n_states=3, buffer_size=K, U_prior=0.8, decay=0.85)

        # Feed K identical deltas so buffer is full
        delta = np.array([0.1, 0.0, 0.0])
        for _ in range(K):
            m.observe(0, delta)

        # U should be mean of buffer (all same normalized value)
        u = m.uncertainty(0)
        assert 0 <= u <= 1

    def test_unvisited_state_unchanged(self):
        """Observing state 0 doesn't affect state 1."""
        m = MetaSR(n_states=3)
        m.observe(0, np.ones(3))
        assert m.uncertainty(1) == m.U_prior

    def test_all_uncertainties_shape(self):
        m = MetaSR(n_states=10)
        U = m.all_uncertainties()
        assert U.shape == (10,)

    def test_all_uncertainties_unvisited(self):
        m = MetaSR(n_states=5, U_prior=0.7)
        U = m.all_uncertainties()
        np.testing.assert_allclose(U, 0.7)


class TestMetaSRConfidence:

    def test_unvisited_low_confidence(self):
        """High uncertainty -> low confidence."""
        m = MetaSR(n_states=3, U_prior=0.8, beta=10.0, theta_C=0.3)
        c = m.confidence(0)
        # U=0.8, theta_C=0.3, beta=10 -> exp(10*(0.8-0.3)) = exp(5) ~ 148
        # C = 1/(1+148) ~ 0.007
        assert c < 0.05

    def test_well_visited_high_confidence(self):
        """After many visits with decreasing errors, confidence should be high."""
        K = 10
        m = MetaSR(n_states=3, buffer_size=K, U_prior=0.8,
                   decay=0.85, beta=10.0, theta_C=0.3)

        # Simulate realistic convergence: large deltas first, then small
        # This creates variance so normalization works properly
        for i in range(K + 20):
            scale = max(0.001, 1.0 / (1 + i))  # decreasing magnitude
            delta = np.array([scale, 0.0, 0.0])
            m.observe(0, delta)

        c = m.confidence(0)
        # Recent deltas are very small -> low U -> high C
        assert c > 0.5

    def test_confidence_in_range(self):
        m = MetaSR(n_states=5)
        for s in range(5):
            c = m.confidence(s)
            assert 0 <= c <= 1

    def test_all_confidences_shape(self):
        m = MetaSR(n_states=10)
        C = m.all_confidences()
        assert C.shape == (10,)

    def test_confidence_inversely_related_to_uncertainty(self):
        """Higher U -> lower C."""
        m = MetaSR(n_states=3, buffer_size=5, U_prior=0.8, beta=10.0, theta_C=0.3)

        # Simulate convergence: decreasing errors create normalization variance
        for i in range(30):
            scale = max(0.001, 1.0 / (1 + i))
            m.observe(0, np.array([scale, 0.0, 0.0]))

        # State 1 never visited -> high U, low C
        c_visited = m.confidence(0)
        c_unvisited = m.confidence(1)
        assert c_visited > c_unvisited


class TestMetaSRChangeDetection:

    def test_no_visits_no_change(self):
        m = MetaSR(n_states=5)
        assert m.detect_change() is False

    def test_initial_visits_detect_change(self):
        """Early on, all states have high U -> change detected."""
        m = MetaSR(n_states=5, U_prior=0.8, theta_change=0.5)
        # Visit some states but U_prior * decay is still > 0.5
        m.observe(0, np.ones(5))
        m.observe(1, np.ones(5))
        # U(0) ~ 0.8*0.85 = 0.68 > 0.5 -> change detected
        assert m.detect_change() is True

    def test_stable_no_change(self):
        """After lots of learning with decreasing errors, no change detected."""
        K = 10
        m = MetaSR(n_states=3, buffer_size=K, U_prior=0.8,
                   decay=0.85, theta_change=0.5)

        # Simulate convergence: decreasing errors over time
        for i in range(60):
            scale = max(0.001, 1.0 / (1 + i))
            for s in range(3):
                m.observe(s, np.array([scale, 0.0, 0.0]))

        assert m.detect_change() is False


class TestMetaSRNormalization:

    def test_normalized_values_bounded(self):
        """Buffer values should be in [0, 1] after normalization."""
        m = MetaSR(n_states=3, buffer_size=5)

        # Feed a mix of large and small deltas
        for _ in range(20):
            m.observe(0, np.random.randn(3) * 10)
            m.observe(0, np.random.randn(3) * 0.01)

        for val in m._buffers[0]:
            assert 0 <= val <= 1.0

    def test_observe_increments_visit_count(self):
        m = MetaSR(n_states=3)
        m.observe(1, np.ones(3))
        m.observe(1, np.ones(3))
        m.observe(1, np.ones(3))
        assert m.visit_counts[1] == 3
        assert m.visit_counts[0] == 0
