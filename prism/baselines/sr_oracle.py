"""SR-Oracle baseline: exploration bonus from true SR error.

Uses ||M(s,:) - M*(s,:)||_2 / sqrt(visits+1) as exploration bonus where
M* is the ground-truth SR computed from the true transition matrix.

The count decay prevents the agent from cycling in high-M* areas where
the raw L2 error is always large (corridor/wall-adjacent cells). Without
it, the spatially smooth error landscape traps the agent in local
attractors (~31% coverage). With count decay, oracle information adds
genuine value beyond pure count-bonus (100% coverage vs 99.8%).

References:
    master.md section 6.2
"""

import numpy as np
from prism.baselines.base_agent import BaseSRAgent


class SROracle(BaseSRAgent):
    """SR agent with oracle exploration bonus based on true SR error.

    bonus(s) = ||M(s,:) - M*(s,:)||_2 / sqrt(visits(s) + 1)

    M* is precomputed from the true transition matrix.
    """

    def __init__(self, n_states, state_mapper, M_star, epsilon=0.1,
                 **kwargs):
        """
        Args:
            M_star: True SR matrix of shape (n_states, n_states).
                    Computed as (I - gamma * T)^{-1}.
        """
        super().__init__(n_states, state_mapper, **kwargs)
        self._M_star = M_star
        self._epsilon = epsilon

    def get_epsilon(self, s, total_steps):
        return self._epsilon

    def exploration_bonus(self, s):
        error = np.linalg.norm(self.sr.M[s] - self._M_star[s])
        return error / np.sqrt(self.visit_counts[s] + 1)
