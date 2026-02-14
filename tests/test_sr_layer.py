"""Tests for SRLayer."""

import numpy as np
import pytest

from prism.agent.sr_layer import SRLayer


class TestSRLayerInit:

    def test_m_initialized_to_identity(self):
        sr = SRLayer(n_states=5)
        np.testing.assert_array_equal(sr.M, np.eye(5))

    def test_r_initialized_to_zeros(self):
        sr = SRLayer(n_states=5)
        np.testing.assert_array_equal(sr.R, np.zeros(5))

    def test_default_params(self):
        sr = SRLayer(n_states=10)
        assert sr.gamma == 0.95
        assert sr.alpha_M == 0.1
        assert sr.alpha_R == 0.3

    def test_custom_params(self):
        sr = SRLayer(n_states=3, gamma=0.9, alpha_M=0.05, alpha_R=0.5)
        assert sr.gamma == 0.9
        assert sr.alpha_M == 0.05
        assert sr.alpha_R == 0.5


class TestSRLayerUpdate:

    def test_update_returns_delta_M(self):
        sr = SRLayer(n_states=5)
        delta = sr.update(s=0, s_next=1, reward=0.0)
        assert delta.shape == (5,)

    def test_delta_M_formula(self):
        """Verify delta_M = e(s') + gamma * M(s',:) - M(s,:) before update."""
        sr = SRLayer(n_states=3, gamma=0.9, alpha_M=0.1, alpha_R=0.3)
        # Before update, M is identity
        # delta = e(1) + 0.9 * M[1,:] - M[0,:]
        # delta = [0,1,0] + 0.9 * [0,1,0] - [1,0,0]
        # delta = [0,1,0] + [0,0.9,0] - [1,0,0] = [-1, 1.9, 0]
        delta = sr.update(s=0, s_next=1, reward=0.0)
        np.testing.assert_allclose(delta, [-1.0, 1.9, 0.0])

    def test_m_updated_after_step(self):
        sr = SRLayer(n_states=3, gamma=0.9, alpha_M=0.1, alpha_R=0.3)
        M_before = sr.M[0].copy()
        delta = sr.update(s=0, s_next=1, reward=0.0)
        expected = M_before + 0.1 * delta
        np.testing.assert_allclose(sr.M[0], expected)

    def test_r_updated_on_reward(self):
        sr = SRLayer(n_states=3, alpha_R=0.3)
        sr.update(s=0, s_next=1, reward=1.0)
        # R[1] = 0 + 0.3 * (1.0 - 0) = 0.3
        assert sr.R[1] == pytest.approx(0.3)

    def test_r_not_updated_without_reward(self):
        sr = SRLayer(n_states=3)
        sr.update(s=0, s_next=1, reward=0.0)
        np.testing.assert_array_equal(sr.R, np.zeros(3))

    def test_only_row_s_changes(self):
        sr = SRLayer(n_states=5)
        M_before = sr.M.copy()
        sr.update(s=2, s_next=3, reward=0.0)
        # Only row 2 should change
        for i in range(5):
            if i != 2:
                np.testing.assert_array_equal(sr.M[i], M_before[i])

    def test_self_transition(self):
        """Staying in the same state: delta should be small."""
        sr = SRLayer(n_states=3, gamma=0.9)
        # delta = e(0) + 0.9*M[0,:] - M[0,:] = e(0) - 0.1*M[0,:]
        # With M=I: delta = [1,0,0] - 0.1*[1,0,0] = [0.9, 0, 0]
        # But actually: e(0) + 0.9*M[0] - M[0] = [1,0,0]+0.9*[1,0,0]-[1,0,0]
        # = [1,0,0]+[0.9,0,0]-[1,0,0] = [0.9,0,0]
        delta = sr.update(s=0, s_next=0, reward=0.0)
        np.testing.assert_allclose(delta, [0.9, 0.0, 0.0])


class TestSRLayerValue:

    def test_value_zero_with_no_reward(self):
        sr = SRLayer(n_states=5)
        assert sr.value(0) == 0.0

    def test_value_after_reward_learning(self):
        sr = SRLayer(n_states=3, alpha_R=1.0)
        # Set R directly: R[1] = 1.0
        sr.update(s=0, s_next=1, reward=1.0)
        # Now R = [0, 1, 0] and M[1,:] has been updated
        # V(1) = M[1,:] . R
        v = sr.value(1)
        assert v > 0  # Should be positive since M[1,1] > 0 and R[1] > 0

    def test_all_values_shape(self):
        sr = SRLayer(n_states=10)
        V = sr.all_values()
        assert V.shape == (10,)

    def test_all_values_consistent_with_value(self):
        sr = SRLayer(n_states=5)
        # Do some updates
        sr.update(0, 1, 0.5)
        sr.update(1, 2, 0.0)
        sr.update(2, 0, 1.0)
        V = sr.all_values()
        for s in range(5):
            assert V[s] == pytest.approx(sr.value(s))


class TestSRLayerConvergence:

    def test_simple_chain_convergence(self):
        """In a simple 3-state chain (0->1->2->0), M should converge."""
        sr = SRLayer(n_states=3, gamma=0.9, alpha_M=0.05, alpha_R=0.3)
        chain = [0, 1, 2]
        for _ in range(3000):
            for i in range(3):
                s = chain[i]
                s_next = chain[(i + 1) % 3]
                sr.update(s, s_next, reward=0.0)

        # After convergence, M[0,1] should be high (0 predicts visiting 1)
        assert sr.M[0, 1] > 0.5
        # M[0,0] should be positive (0 predicts revisiting itself via chain)
        assert sr.M[0, 0] > 0.3

    def test_reward_value_propagation(self):
        """Reward at state 2 should propagate value to earlier states."""
        sr = SRLayer(n_states=3, gamma=0.9, alpha_M=0.05, alpha_R=0.3)
        for _ in range(2000):
            sr.update(0, 1, reward=0.0)
            sr.update(1, 2, reward=1.0)
            sr.update(2, 0, reward=0.0)

        # V(1) should be highest (closest to reward at state 2)
        V = sr.all_values()
        assert V[1] > V[0]
        # All values should be positive (reward propagates through chain)
        assert all(v > 0 for v in V)
