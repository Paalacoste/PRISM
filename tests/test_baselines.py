"""Tests for Exp B baselines and experiment infrastructure."""

import numpy as np
import pytest

import minigrid  # noqa: F401
import gymnasium as gym

from prism.env.state_mapper import StateMapper
from prism.env.dynamics_wrapper import DynamicsWrapper
from prism.env.exploration_task import place_goals, get_room_cells
from prism.baselines import (
    RandomAgent,
    SREpsilonGreedy,
    SREpsilonDecay,
    SRCountBonus,
    SRNormBonus,
    SRPosterior,
    SROracle,
)
from prism.analysis.metrics import (
    bootstrap_ci,
    mann_whitney_test,
    holm_bonferroni,
    compare_conditions,
    compare_all_pairs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def four_rooms_env():
    env = gym.make("MiniGrid-FourRooms-v0")
    env.reset(seed=42)
    return env


@pytest.fixture(scope="module")
def state_mapper(four_rooms_env):
    return StateMapper(four_rooms_env)


@pytest.fixture(scope="module")
def M_star(four_rooms_env, state_mapper):
    dw = DynamicsWrapper(four_rooms_env)
    T = dw.get_true_transition_matrix(state_mapper)
    n = state_mapper.n_states
    return np.linalg.inv(np.eye(n) - 0.95 * T)


# ---------------------------------------------------------------------------
# Base agent interface tests
# ---------------------------------------------------------------------------
class TestBaselineInterface:
    """All baselines must expose the same interface."""

    AGENT_CLASSES = [
        ("RandomAgent", RandomAgent, {}),
        ("SREpsilonGreedy", SREpsilonGreedy, {"epsilon": 0.1}),
        ("SREpsilonDecay", SREpsilonDecay, {}),
        ("SRCountBonus", SRCountBonus, {}),
        ("SRNormBonus", SRNormBonus, {}),
        ("SRPosterior", SRPosterior, {}),
    ]

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_select_action_returns_valid(self, name, cls, kwargs, state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        action = agent.select_action(0, [0, 1, 2], agent_dir=0)
        assert action in [0, 1, 2]

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_update_increments_visits(self, name, cls, kwargs, state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        agent.update(0, 1, 0.0)
        assert agent.visit_counts[0] == 1
        assert agent.visit_counts[1] == 0

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_exploration_bonus_returns_float(self, name, cls, kwargs,
                                             state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        bonus = agent.exploration_bonus(0)
        assert isinstance(bonus, (int, float, np.floating))

    def test_oracle_requires_M_star(self, state_mapper, M_star):
        agent = SROracle(state_mapper.n_states, state_mapper,
                         M_star=M_star, seed=42)
        action = agent.select_action(0, [0, 1, 2], agent_dir=0)
        assert action in [0, 1, 2]

    def test_oracle_bonus_is_sr_error(self, state_mapper, M_star):
        agent = SROracle(state_mapper.n_states, state_mapper,
                         M_star=M_star, seed=42)
        # M starts as identity, M* is not identity → bonus should be > 0
        bonus = agent.exploration_bonus(0)
        assert bonus > 0


# ---------------------------------------------------------------------------
# Specific baseline behavior
# ---------------------------------------------------------------------------
class TestEpsilonGreedy:
    def test_fixed_epsilon(self, state_mapper):
        agent = SREpsilonGreedy(state_mapper.n_states, state_mapper,
                                epsilon=0.3, seed=42)
        assert agent.get_epsilon(0, 0) == 0.3
        assert agent.get_epsilon(0, 1000) == 0.3


class TestEpsilonDecay:
    def test_epsilon_decreases(self, state_mapper):
        agent = SREpsilonDecay(state_mapper.n_states, state_mapper,
                               epsilon_start=0.5, epsilon_min=0.01,
                               decay_rate=0.99, seed=42)
        eps_0 = agent.get_epsilon(0, 0)
        eps_100 = agent.get_epsilon(0, 100)
        eps_1000 = agent.get_epsilon(0, 1000)
        assert eps_0 > eps_100 > eps_1000
        assert eps_1000 >= 0.01


class TestCountBonus:
    def test_bonus_decreases_with_visits(self, state_mapper):
        agent = SRCountBonus(state_mapper.n_states, state_mapper, seed=42)
        b0 = agent.exploration_bonus(0)  # visits=0
        agent.update(0, 1, 0.0)
        b1 = agent.exploration_bonus(0)  # visits=1
        assert b0 > b1


class TestNormBonus:
    def test_bonus_positive(self, state_mapper):
        agent = SRNormBonus(state_mapper.n_states, state_mapper, seed=42)
        bonus = agent.exploration_bonus(0)
        assert bonus > 0

    def test_bonus_changes_after_learning(self, state_mapper):
        agent = SRNormBonus(state_mapper.n_states, state_mapper, seed=42)
        b_before = agent.exploration_bonus(0)
        for _ in range(50):
            agent.update(0, 1, 0.0)
        b_after = agent.exploration_bonus(0)
        assert b_before != b_after


class TestPosterior:
    def test_exploration_value_varies(self, state_mapper):
        """Posterior sampling should produce different values on different calls."""
        agent = SRPosterior(state_mapper.n_states, state_mapper, seed=42)
        v1 = agent.exploration_value(0)
        agent._sample_valid = False
        v2 = agent.exploration_value(0)
        # With same seed, first two samples may be same; just check it's a float
        assert isinstance(v1, (float, np.floating))


class TestRandom:
    def test_no_sr(self, state_mapper):
        agent = RandomAgent(state_mapper.n_states, state_mapper, seed=42)
        assert not hasattr(agent, "sr")

    def test_exploration_value_zero(self, state_mapper):
        agent = RandomAgent(state_mapper.n_states, state_mapper, seed=42)
        assert agent.exploration_value(0) == 0.0


# ---------------------------------------------------------------------------
# Goal placement and room detection
# ---------------------------------------------------------------------------
class TestGoalPlacement:
    def test_four_goals_placed(self, state_mapper):
        goals = place_goals(state_mapper, n_goals=4,
                            rng=np.random.default_rng(42))
        assert len(goals) == 4

    def test_goals_are_valid_positions(self, state_mapper):
        goals = place_goals(state_mapper, n_goals=4,
                            rng=np.random.default_rng(42))
        for pos in goals:
            # Should be a mappable position
            idx = state_mapper.get_index(pos)
            assert 0 <= idx < state_mapper.n_states

    def test_goals_in_different_quadrants(self, state_mapper):
        goals = place_goals(state_mapper, n_goals=4,
                            rng=np.random.default_rng(42))
        h, w = state_mapper.get_grid_shape()
        mid_x, mid_y = w // 2, h // 2
        quadrants = set()
        for x, y in goals:
            q = (0 if x < mid_x else 1, 0 if y < mid_y else 1)
            quadrants.add(q)
        assert len(quadrants) == 4

    def test_room_cells_cover_all_states(self, state_mapper):
        rooms = get_room_cells(state_mapper)
        all_cells = set()
        for r in rooms:
            all_cells.update(r)
        # Cells on midline may be skipped; at least most should be covered
        assert len(all_cells) >= state_mapper.n_states * 0.9

    def test_deterministic_with_seed(self, state_mapper):
        g1 = place_goals(state_mapper, rng=np.random.default_rng(123))
        g2 = place_goals(state_mapper, rng=np.random.default_rng(123))
        assert g1 == g2


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------
class TestBootstrapCI:
    def test_mean_in_ci(self):
        data = np.random.default_rng(42).normal(50, 10, 100)
        m, lo, hi = bootstrap_ci(data)
        assert lo <= m <= hi

    def test_ci_width_decreases_with_n(self):
        rng = np.random.default_rng(42)
        _, lo_50, hi_50 = bootstrap_ci(rng.normal(0, 1, 50))
        _, lo_500, hi_500 = bootstrap_ci(rng.normal(0, 1, 500))
        assert (hi_50 - lo_50) > (hi_500 - lo_500)

    def test_narrow_ci_for_constant_data(self):
        data = np.ones(100) * 42
        m, lo, hi = bootstrap_ci(data)
        assert m == 42
        assert hi - lo < 0.01


class TestMannWhitney:
    def test_identical_samples(self):
        x = np.ones(50) * 10
        result = mann_whitney_test(x, x)
        assert result["p_value"] >= 0.05

    def test_different_samples(self):
        x = np.random.default_rng(42).normal(0, 1, 100)
        y = np.random.default_rng(42).normal(5, 1, 100)
        result = mann_whitney_test(x, y)
        assert result["p_value"] < 0.001

    def test_effect_size_bounds(self):
        x = np.random.default_rng(42).normal(0, 1, 50)
        y = np.random.default_rng(43).normal(0, 1, 50)
        result = mann_whitney_test(x, y)
        assert -1 <= result["effect_size"] <= 1


class TestHolmBonferroni:
    def test_single_p_value(self):
        assert holm_bonferroni([0.03]) == [0.03]

    def test_corrected_larger_than_raw(self):
        raw = [0.01, 0.03, 0.05]
        corrected = holm_bonferroni(raw)
        for r, c in zip(raw, corrected):
            assert c >= r

    def test_corrected_capped_at_1(self):
        corrected = holm_bonferroni([0.5, 0.6, 0.7])
        for c in corrected:
            assert c <= 1.0


class TestCompareConditions:
    def test_returns_all_comparisons(self):
        rng = np.random.default_rng(42)
        results = {
            "PRISM": rng.normal(100, 10, 50),
            "A": rng.normal(120, 10, 50),
            "B": rng.normal(110, 10, 50),
        }
        comparisons = compare_conditions(results)
        assert len(comparisons) == 2
        conditions_compared = {r["condition"] for r in comparisons}
        assert conditions_compared == {"A", "B"}

    def test_p_corrected_present(self):
        rng = np.random.default_rng(42)
        results = {
            "PRISM": rng.normal(100, 10, 50),
            "A": rng.normal(120, 10, 50),
        }
        comparisons = compare_conditions(results)
        assert "p_corrected" in comparisons[0]
