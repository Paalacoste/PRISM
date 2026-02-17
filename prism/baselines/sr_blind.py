"""SR agents with no exploration bonus: epsilon-greedy and epsilon-decay."""

from prism.baselines.base_agent import BaseSRAgent


class SREpsilonGreedy(BaseSRAgent):
    """SR agent with fixed epsilon-greedy exploration. No bonus."""

    def __init__(self, n_states, state_mapper, epsilon=0.1, **kwargs):
        super().__init__(n_states, state_mapper, lambda_explore=0.0, **kwargs)
        self._epsilon = epsilon

    def get_epsilon(self, s, total_steps):
        return self._epsilon


class SREpsilonDecay(BaseSRAgent):
    """SR agent with decaying epsilon. No bonus.

    epsilon(t) = max(epsilon_min, epsilon_start * decay_rate^t)
    """

    def __init__(self, n_states, state_mapper, epsilon_start=0.5,
                 epsilon_min=0.01, decay_rate=0.999, **kwargs):
        super().__init__(n_states, state_mapper, lambda_explore=0.0, **kwargs)
        self._epsilon_start = epsilon_start
        self._epsilon_min = epsilon_min
        self._decay_rate = decay_rate

    def get_epsilon(self, s, total_steps):
        return max(self._epsilon_min,
                   self._epsilon_start * self._decay_rate ** total_steps)
