"""Base agent class for Exp B baselines.

All SR-based baselines share:
- A tabular SRLayer for learning transitions
- MiniGrid-aware action selection (turn/forward with 2-step lookahead)
- Configurable epsilon schedule and exploration bonus

Subclasses override get_epsilon() and exploration_bonus().
"""

import numpy as np
from prism.agent.sr_layer import SRLayer

# MiniGrid direction vectors: 0=right, 1=down, 2=left, 3=up
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# MiniGrid movement actions
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2


class BaseSRAgent:
    """Base SR agent with configurable exploration strategy.

    Handles SR learning, MiniGrid navigation (turn/forward),
    and epsilon-greedy action selection with exploration bonus.

    Subclasses must override:
        get_epsilon(s, total_steps) -> float
        exploration_bonus(s) -> float
    """

    def __init__(self, n_states, state_mapper, gamma=0.95, alpha_M=0.1,
                 alpha_R=0.3, lambda_explore=0.5, seed=42):
        self.n_states = n_states
        self.mapper = state_mapper
        self.lambda_explore = lambda_explore
        self.sr = SRLayer(n_states, gamma, alpha_M, alpha_R)
        self.visit_counts = np.zeros(n_states, dtype=np.int64)
        self.total_steps = 0
        self._rng = np.random.default_rng(seed)

    def get_epsilon(self, s: int, total_steps: int) -> float:
        """Return current exploration rate. Override in subclasses."""
        return 0.1

    def exploration_bonus(self, s: int) -> float:
        """Return exploration bonus for state s. Override in subclasses."""
        return 0.0

    def exploration_value(self, s: int) -> float:
        """V_explore(s) = V(s) + lambda * bonus(s)."""
        return self.sr.value(s) + self.lambda_explore * self.exploration_bonus(s)

    def update(self, s: int, s_next: int, reward: float):
        """Update SR and visit counts."""
        self.sr.update(s, s_next, reward)
        self.visit_counts[s] += 1
        self.total_steps += 1

    def select_action(self, s: int, available_actions: list[int],
                      agent_dir: int) -> int:
        """Epsilon-greedy action selection with V_explore."""
        epsilon = self.get_epsilon(s, self.total_steps)

        if self._rng.random() < epsilon:
            return int(self._rng.choice(available_actions))
        return self._greedy_action(s, agent_dir, available_actions)

    def _greedy_action(self, s: int, agent_dir: int,
                       available_actions: list[int]) -> int:
        """Pick action leading to highest V_explore neighbor."""
        pos = self.mapper.get_pos(s)

        # Evaluate V_explore for each accessible neighbor direction
        candidates = []
        for d in range(4):
            dx, dy = DIR_VEC[d]
            nx, ny = pos[0] + dx, pos[1] + dy
            try:
                s_neighbor = self.mapper.get_index((nx, ny))
                v = self.exploration_value(s_neighbor)
                candidates.append((d, v))
            except KeyError:
                pass  # wall

        if not candidates:
            return int(self._rng.choice(available_actions))

        # Best direction with random tie-breaking
        best_value = max(v for _, v in candidates)
        best_dirs = [d for d, v in candidates if np.isclose(v, best_value)]
        best_dir = int(self._rng.choice(best_dirs))

        return self._dir_to_action(agent_dir, best_dir)

    @staticmethod
    def _dir_to_action(current_dir: int, target_dir: int) -> int:
        """Convert desired facing direction to a MiniGrid action."""
        if target_dir == current_dir:
            return ACTION_FORWARD
        if (current_dir - 1) % 4 == target_dir:
            return ACTION_LEFT
        if (current_dir + 1) % 4 == target_dir:
            return ACTION_RIGHT
        # 180 turn — turn left (arbitrary)
        return ACTION_LEFT


class RandomAgent:
    """Uniform random action selection. No learning."""

    def __init__(self, n_states, state_mapper, seed=42, **kwargs):
        self.n_states = n_states
        self.mapper = state_mapper
        self.visit_counts = np.zeros(n_states, dtype=np.int64)
        self.total_steps = 0
        self._rng = np.random.default_rng(seed)

    def select_action(self, s: int, available_actions: list[int],
                      agent_dir: int) -> int:
        return int(self._rng.choice(available_actions))

    def update(self, s: int, s_next: int, reward: float):
        self.visit_counts[s] += 1
        self.total_steps += 1

    def exploration_bonus(self, s: int) -> float:
        return 0.0

    def exploration_value(self, s: int) -> float:
        return 0.0
