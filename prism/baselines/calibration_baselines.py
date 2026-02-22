"""Calibration baselines for Experiment A.

4 agent types providing confidence scores for calibration comparison:
- SRGlobalConf: single global confidence (no per-state structure)
- SRCountConf: count-based confidence C(s) = 1 - 1/sqrt(visits+1)
- SRBayesianConf: Bayesian posterior variance on M rows
- RandomConfAgent: random confidence in [0,1]

All expose confidence(s), all_confidences(), all_uncertainties()
for compatibility with the Exp A calibration metrics.
"""

from collections import deque

import numpy as np

from prism.agent.sr_layer import SRLayer
from prism.baselines.base_agent import BaseSRAgent


class SRGlobalConf(BaseSRAgent):
    """SR agent with a single global confidence signal.

    Confidence is uniform across all states, based on the global
    mean of recent TD error norms. Tests whether per-state
    localization matters for calibration.
    """

    def __init__(self, n_states, state_mapper, epsilon=0.1,
                 beta=10.0, theta=0.3, buffer_size=500, **kwargs):
        super().__init__(n_states, state_mapper, **kwargs)
        self._epsilon = epsilon
        self._beta = beta
        self._theta = theta
        self._delta_buffer = deque(maxlen=buffer_size)
        self._global_U = 0.8  # prior

    def get_epsilon(self, s, total_steps):
        return self._epsilon

    def exploration_bonus(self, s):
        return self._global_U

    def update(self, s, s_next, reward):
        delta_M = self.sr.update(s, s_next, reward)
        self.visit_counts[s] += 1
        self.total_steps += 1
        delta_scalar = float(np.linalg.norm(delta_M))
        self._delta_buffer.append(delta_scalar)
        if len(self._delta_buffer) >= 10:
            self._global_U = float(np.mean(self._delta_buffer))

    def uncertainty(self, s):
        return self._global_U

    def all_uncertainties(self):
        return np.full(self.n_states, self._global_U)

    def confidence(self, s):
        return 1.0 / (1.0 + np.exp(self._beta * (self._global_U - self._theta)))

    def all_confidences(self):
        c = self.confidence(0)
        return np.full(self.n_states, c)


class SRCountConf(BaseSRAgent):
    """SR agent with count-based confidence.

    C(s) = 1 - 1/sqrt(visits(s) + 1)
    U(s) = 1/sqrt(visits(s) + 1)

    Has per-state structure but ignores SR prediction quality.
    """

    def __init__(self, n_states, state_mapper, epsilon=0.1, **kwargs):
        super().__init__(n_states, state_mapper, **kwargs)
        self._epsilon = epsilon

    def get_epsilon(self, s, total_steps):
        return self._epsilon

    def uncertainty(self, s):
        return 1.0 / np.sqrt(self.visit_counts[s] + 1)

    def all_uncertainties(self):
        return 1.0 / np.sqrt(self.visit_counts + 1)

    def confidence(self, s):
        return 1.0 - self.uncertainty(s)

    def all_confidences(self):
        return 1.0 - self.all_uncertainties()


class SRBayesianConf(BaseSRAgent):
    """SR agent with Bayesian confidence based on posterior variance.

    Uses Welford's online algorithm to track variance of TD error
    norms per state. Posterior uncertainty shrinks with observations
    and increases when predictions are noisy.
    """

    def __init__(self, n_states, state_mapper, epsilon=0.1,
                 beta=10.0, theta=0.3, U_prior=0.8, **kwargs):
        super().__init__(n_states, state_mapper, **kwargs)
        self._epsilon = epsilon
        self._beta = beta
        self._theta = theta
        self._U_prior = U_prior
        # Welford online variance per state
        self._mean_delta = np.zeros(n_states)
        self._M2_delta = np.zeros(n_states)
        self._delta_count = np.zeros(n_states)

    def get_epsilon(self, s, total_steps):
        return self._epsilon

    def exploration_bonus(self, s):
        return self.uncertainty(s)

    def update(self, s, s_next, reward):
        delta_M = self.sr.update(s, s_next, reward)
        self.visit_counts[s] += 1
        self.total_steps += 1
        # Welford online algorithm
        delta_scalar = float(np.linalg.norm(delta_M))
        self._delta_count[s] += 1
        n = self._delta_count[s]
        d1 = delta_scalar - self._mean_delta[s]
        self._mean_delta[s] += d1 / n
        d2 = delta_scalar - self._mean_delta[s]
        self._M2_delta[s] += d1 * d2

    def uncertainty(self, s):
        n = self._delta_count[s]
        if n < 2:
            return self._U_prior
        var = self._M2_delta[s] / (n - 1)
        return float(var / np.sqrt(n))

    def all_uncertainties(self):
        return np.array([self.uncertainty(s) for s in range(self.n_states)])

    def confidence(self, s):
        U = self.uncertainty(s)
        return 1.0 / (1.0 + np.exp(self._beta * (U - self._theta)))

    def all_confidences(self):
        U = self.all_uncertainties()
        return 1.0 / (1.0 + np.exp(self._beta * (U - self._theta)))


class RandomConfAgent:
    """Agent with random confidence scores. No learning of confidence.

    Learns M via SRLayer (for fair M vs M* comparison) but confidence
    is drawn from Uniform[0, 1], fixed per state at construction.
    """

    def __init__(self, n_states, state_mapper, seed=42, **kwargs):
        self.n_states = n_states
        self.mapper = state_mapper
        self.visit_counts = np.zeros(n_states, dtype=np.int64)
        self.total_steps = 0
        self._rng = np.random.default_rng(seed)
        self.sr = SRLayer(n_states)
        self._confidences = self._rng.random(n_states)

    def select_action(self, s, available_actions, agent_dir):
        return int(self._rng.choice(available_actions))

    def update(self, s, s_next, reward):
        self.sr.update(s, s_next, reward)
        self.visit_counts[s] += 1
        self.total_steps += 1

    def uncertainty(self, s):
        return 1.0 - self._confidences[s]

    def all_uncertainties(self):
        return 1.0 - self._confidences

    def confidence(self, s):
        return float(self._confidences[s])

    def all_confidences(self):
        return self._confidences.copy()

    def reseed_confidence(self, seed):
        """Re-randomize confidence scores."""
        self._rng = np.random.default_rng(seed)
        self._confidences = self._rng.random(self.n_states)

    def exploration_bonus(self, s):
        return 0.0

    def exploration_value(self, s):
        return 0.0
