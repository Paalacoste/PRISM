"""Tests for Exp A calibration baselines and PerturbationSchedule."""

import numpy as np
import pytest
import minigrid  # noqa: F401 — register envs
import gymnasium as gym

from prism.env.state_mapper import StateMapper
from prism.env.perturbation_schedule import PerturbationSchedule, PerturbationEvent
from prism.baselines.calibration_baselines import (
    SRGlobalConf,
    SRCountConf,
    SRBayesianConf,
    RandomConfAgent,
)


@pytest.fixture(scope="module")
def four_rooms_env():
    env = gym.make("MiniGrid-FourRooms-v0")
    env.reset(seed=42)
    return env


@pytest.fixture(scope="module")
def state_mapper(four_rooms_env):
    return StateMapper(four_rooms_env)


# ── PerturbationSchedule ─────────────────────────────────────────────

class TestPerturbationSchedule:

    def test_init_sorts_events(self):
        events = [
            PerturbationEvent(200, "reward_shift", {"new_goal_pos": (5, 5)}),
            PerturbationEvent(100, "reward_shift", {"new_goal_pos": (3, 3)}),
        ]
        sched = PerturbationSchedule(events)
        assert sched.events[0].episode == 100
        assert sched.events[1].episode == 200

    def test_get_events_for_episode(self):
        events = [
            PerturbationEvent(100, "reward_shift", {"new_goal_pos": (3, 3)}),
            PerturbationEvent(200, "reward_shift", {"new_goal_pos": (5, 5)}),
        ]
        sched = PerturbationSchedule(events)
        assert len(sched.get_events_for_episode(50)) == 0
        assert len(sched.get_events_for_episode(100)) == 1
        assert len(sched.get_events_for_episode(200)) == 1

    def test_reset(self):
        events = [PerturbationEvent(10, "reward_shift")]
        sched = PerturbationSchedule(events)
        sched.get_events_for_episode(10)
        assert len(sched.get_events_for_episode(10)) == 0  # already consumed
        sched.reset()
        assert len(sched.get_events_for_episode(10)) == 1  # re-triggers

    def test_phase_boundaries(self):
        goals = [(1, 1), (5, 5), (10, 10)]
        sched = PerturbationSchedule.exp_a(goal_positions=goals)
        assert sched.phase_boundaries() == [200, 400]

    def test_exp_a_creates_2_events(self):
        goals = [(1, 1), (5, 5), (10, 10)]
        sched = PerturbationSchedule.exp_a(
            phase_1_episodes=200, phase_2_episodes=200,
            goal_positions=goals,
        )
        assert len(sched.events) == 2
        assert sched.events[0].episode == 200
        assert sched.events[0].ptype == "reward_shift"
        assert sched.events[1].episode == 400

    def test_exp_a_custom_lengths(self):
        goals = [(1, 1), (5, 5), (10, 10)]
        sched = PerturbationSchedule.exp_a(
            phase_1_episodes=100, phase_2_episodes=150,
            goal_positions=goals,
        )
        assert sched.events[0].episode == 100
        assert sched.events[1].episode == 250

    def test_exp_a_requires_3_goals(self):
        with pytest.raises(ValueError):
            PerturbationSchedule.exp_a(goal_positions=[(1, 1)])

    def test_exp_a_none_goals_raises(self):
        with pytest.raises(ValueError):
            PerturbationSchedule.exp_a(goal_positions=None)

    def test_exp_c_not_implemented(self):
        with pytest.raises(NotImplementedError):
            PerturbationSchedule.exp_c()

    def test_default_kwargs(self):
        e = PerturbationEvent(10, "reward_shift")
        assert e.kwargs == {}


# ── Calibration Baselines Interface ──────────────────────────────────

AGENT_CLASSES = [
    ("SRGlobalConf", SRGlobalConf, {}),
    ("SRCountConf", SRCountConf, {}),
    ("SRBayesianConf", SRBayesianConf, {}),
    ("RandomConfAgent", RandomConfAgent, {}),
]


class TestCalibrationBaselineInterface:

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_confidence_in_range(self, name, cls, kwargs, state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        c = agent.confidence(0)
        assert 0 <= c <= 1

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_all_confidences_shape(self, name, cls, kwargs, state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        confs = agent.all_confidences()
        assert confs.shape == (state_mapper.n_states,)
        assert np.all(confs >= 0) and np.all(confs <= 1)

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_all_uncertainties_shape(self, name, cls, kwargs, state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        uncs = agent.all_uncertainties()
        assert uncs.shape == (state_mapper.n_states,)

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_has_sr_M(self, name, cls, kwargs, state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        assert hasattr(agent, "sr")
        n = state_mapper.n_states
        assert agent.sr.M.shape == (n, n)

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_select_action_valid(self, name, cls, kwargs, state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        action = agent.select_action(0, [0, 1, 2], agent_dir=0)
        assert action in [0, 1, 2]

    @pytest.mark.parametrize("name,cls,kwargs", AGENT_CLASSES,
                             ids=[a[0] for a in AGENT_CLASSES])
    def test_update_increments_visits(self, name, cls, kwargs, state_mapper):
        agent = cls(state_mapper.n_states, state_mapper, seed=42, **kwargs)
        agent.update(0, 1, 0.0)
        assert agent.visit_counts[0] == 1


# ── SRGlobalConf ─────────────────────────────────────────────────────

class TestSRGlobalConf:

    def test_uniform_confidence(self, state_mapper):
        agent = SRGlobalConf(state_mapper.n_states, state_mapper, seed=42)
        confs = agent.all_confidences()
        assert np.all(confs == confs[0])

    def test_confidence_changes_with_training(self, state_mapper):
        agent = SRGlobalConf(state_mapper.n_states, state_mapper, seed=42)
        c_before = agent.confidence(0)
        for i in range(50):
            agent.update(i % 10, (i + 1) % 10, 0.0)
        c_after = agent.confidence(0)
        assert c_before != c_after


# ── SRCountConf ──────────────────────────────────────────────────────

class TestSRCountConf:

    def test_confidence_increases_with_visits(self, state_mapper):
        agent = SRCountConf(state_mapper.n_states, state_mapper, seed=42)
        c0 = agent.confidence(0)
        agent.update(0, 1, 0.0)
        c1 = agent.confidence(0)
        assert c1 > c0

    def test_unvisited_zero_confidence(self, state_mapper):
        agent = SRCountConf(state_mapper.n_states, state_mapper, seed=42)
        # visits=0 -> U=1/sqrt(1)=1 -> C=0
        assert agent.confidence(100) == 0.0


# ── SRBayesianConf ───────────────────────────────────────────────────

class TestSRBayesianConf:

    def test_prior_uncertainty_for_unvisited(self, state_mapper):
        agent = SRBayesianConf(state_mapper.n_states, state_mapper, seed=42)
        assert agent.uncertainty(0) == 0.8

    def test_uncertainty_changes_after_training(self, state_mapper):
        agent = SRBayesianConf(state_mapper.n_states, state_mapper, seed=42)
        u_before = agent.uncertainty(0)
        for _ in range(30):
            agent.update(0, 1, 0.0)
        u_after = agent.uncertainty(0)
        assert u_after != u_before


# ── RandomConfAgent ──────────────────────────────────────────────────

class TestRandomConfAgent:

    def test_random_confidences_have_variance(self, state_mapper):
        agent = RandomConfAgent(state_mapper.n_states, state_mapper, seed=42)
        confs = agent.all_confidences()
        assert np.std(confs) > 0.1

    def test_reseed_changes_confidences(self, state_mapper):
        agent = RandomConfAgent(state_mapper.n_states, state_mapper, seed=42)
        c_before = agent.all_confidences().copy()
        agent.reseed_confidence(99)
        c_after = agent.all_confidences()
        assert not np.allclose(c_before, c_after)

    def test_learns_sr(self, state_mapper):
        agent = RandomConfAgent(state_mapper.n_states, state_mapper, seed=42)
        M_before = agent.sr.M.copy()
        agent.update(0, 1, 0.0)
        assert not np.allclose(agent.sr.M, M_before)

    def test_exploration_bonus_zero(self, state_mapper):
        agent = RandomConfAgent(state_mapper.n_states, state_mapper, seed=42)
        assert agent.exploration_bonus(0) == 0.0
