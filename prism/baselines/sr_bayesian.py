"""SR agent with posterior sampling (Thompson-style exploration).

Instead of epsilon-greedy + bonus, samples V from a posterior
and acts greedily on the sample. This naturally balances
exploration and exploitation via uncertainty in V.

References:
    Janz et al. (2019) — Successor Uncertainties.
"""

import numpy as np
from prism.baselines.base_agent import BaseSRAgent


class SRPosterior(BaseSRAgent):
    """SR agent with posterior sampling on V.

    Maintains a diagonal Gaussian posterior over R (reward vector).
    At each step, samples R ~ N(mu_R, sigma_R^2) and acts greedily
    on V_sample = M @ R_sample.

    The posterior variance shrinks with observations, providing
    natural exploration bonuses via optimistic samples.
    """

    def __init__(self, n_states, state_mapper, sigma_prior=1.0,
                 epsilon=0.05, **kwargs):
        super().__init__(n_states, state_mapper, lambda_explore=0.0, **kwargs)
        self._epsilon = epsilon
        # Posterior parameters for R: independent Gaussian per state
        self._R_mu = np.zeros(n_states)
        self._R_var = np.full(n_states, sigma_prior ** 2)
        self._R_n = np.zeros(n_states)  # observation count per state
        self._sigma_prior = sigma_prior
        # Cache current V sample for one step
        self._v_sample = np.zeros(n_states)
        self._sample_valid = False

    def get_epsilon(self, s, total_steps):
        return self._epsilon

    def update(self, s, s_next, reward):
        super().update(s, s_next, reward)
        # Bayesian update of R posterior at s_next
        self._R_n[s_next] += 1
        n = self._R_n[s_next]
        prior_var = self._sigma_prior ** 2
        # Running mean with prior shrinkage on variance
        self._R_mu[s_next] = (self._R_mu[s_next] * (n - 1) + reward) / n
        self._R_var[s_next] = prior_var / (1 + n)
        self._sample_valid = False

    def _resample_v(self):
        """Sample V = M @ R_sample where R_sample ~ posterior."""
        R_sample = self._rng.normal(self._R_mu, np.sqrt(self._R_var))
        self._v_sample = self.sr.M @ R_sample
        self._sample_valid = True

    def exploration_value(self, s):
        """Use sampled V instead of point estimate."""
        if not self._sample_valid:
            self._resample_v()
        return self._v_sample[s]
