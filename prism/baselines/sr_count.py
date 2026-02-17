"""SR agents with count-based and norm-based exploration bonuses."""

import numpy as np
from prism.baselines.base_agent import BaseSRAgent


class SRCountBonus(BaseSRAgent):
    """SR agent with count-based exploration bonus: lambda / sqrt(visits+1).

    References:
        Machado et al. (2020) — count-based exploration with SR.
    """

    def __init__(self, n_states, state_mapper, epsilon=0.1, **kwargs):
        super().__init__(n_states, state_mapper, **kwargs)
        self._epsilon = epsilon

    def get_epsilon(self, s, total_steps):
        return self._epsilon

    def exploration_bonus(self, s):
        return 1.0 / np.sqrt(self.visit_counts[s] + 1)


class SRNormBonus(BaseSRAgent):
    """SR agent with SR-norm exploration bonus: lambda / (||M(s,:)|| + eps).

    Low SR norm = state is hard to reach = explore it more.

    References:
        Machado et al. (2020) — SR norm as novelty signal.
    """

    def __init__(self, n_states, state_mapper, epsilon=0.1, **kwargs):
        super().__init__(n_states, state_mapper, **kwargs)
        self._epsilon = epsilon

    def get_epsilon(self, s, total_steps):
        return self._epsilon

    def exploration_bonus(self, s):
        norm = np.linalg.norm(self.sr.M[s])
        return 1.0 / (norm + 1e-8)
