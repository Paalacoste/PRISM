"""Adaptive controller using meta-SR signals for action selection.

Uses U(s) for adaptive exploration rate and exploration bonus.
V_explore(s) = V(s) + lambda * U(s)
epsilon_adaptive(s) = epsilon_min + (epsilon_max - epsilon_min) * U(s)

References:
    master.md section 5.4
"""

import numpy as np


class PRISMController:
    """Policy controller integrating SR values with meta-SR uncertainty.

    Action selection:
        1. Compute exploration value V_explore for neighboring states
        2. With probability epsilon_adaptive(s): random action
        3. Otherwise: greedy on V_explore
        4. Report confidence and "I don't know" flag
    """

    def __init__(self, sr_layer, meta_sr, state_mapper,
                 epsilon_min: float = 0.01, epsilon_max: float = 0.5,
                 lambda_explore: float = 0.5, theta_idk: float = 0.3):
        """Initialize the controller.

        Args:
            sr_layer: SRLayer instance.
            meta_sr: MetaSR instance.
            state_mapper: StateMapper instance.
            epsilon_min: Exploration floor.
            epsilon_max: Exploration ceiling.
            lambda_explore: Weight of uncertainty bonus in V_explore.
            theta_idk: Confidence threshold for "I don't know" signal.
        """
        self.sr = sr_layer
        self.meta_sr = meta_sr
        self.mapper = state_mapper
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.lambda_explore = lambda_explore
        self.theta_idk = theta_idk
        self._rng = np.random.default_rng()

    def seed(self, seed: int):
        """Set the random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def adaptive_epsilon(self, s: int) -> float:
        """Compute exploration rate based on uncertainty at state s.

        epsilon(s) = epsilon_min + (epsilon_max - epsilon_min) * U(s)
        """
        U_s = self.meta_sr.uncertainty(s)
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * U_s

    def exploration_value(self, s: int) -> float:
        """Compute V_explore(s) = V(s) + lambda * U(s)."""
        V_s = self.sr.value(s)
        U_s = self.meta_sr.uncertainty(s)
        return V_s + self.lambda_explore * U_s

    def select_action(self, s: int, available_actions: list[int],
                      agent_dir: int | None = None) -> tuple[int, float, bool]:
        """Select an action using adaptive epsilon-greedy on V_explore.

        Args:
            s: Current state index.
            available_actions: List of valid action indices.
            agent_dir: Agent's current facing direction (0-3). Required for
                       greedy action selection in MiniGrid.

        Returns:
            Tuple of (action, confidence, idk_flag):
                action: Chosen action index.
                confidence: C(s) in [0, 1].
                idk_flag: True if C(s) < theta_idk.
        """
        confidence = self.meta_sr.confidence(s)
        idk_flag = confidence < self.theta_idk
        epsilon = self.adaptive_epsilon(s)

        if self._rng.random() < epsilon:
            action = self._rng.choice(available_actions)
        elif agent_dir is not None:
            action = self._greedy_v_explore(s, agent_dir, available_actions)
        else:
            action = self._rng.choice(available_actions)

        return (int(action), float(confidence), bool(idk_flag))

    def _greedy_v_explore(self, s: int, agent_dir: int,
                          available_actions: list[int]) -> int:
        """Pick action leading to the highest V_explore neighbor.

        Evaluates V_explore for each of the 4 cardinal neighbors,
        then returns the MiniGrid action to move towards the best one.
        """
        # MiniGrid direction vectors: 0=right, 1=down, 2=left, 3=up
        dir_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        pos = self.mapper.get_pos(s)

        candidates = []
        for d in range(4):
            dx, dy = dir_vec[d]
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

        # Convert desired direction to MiniGrid action
        if best_dir == agent_dir:
            return 2  # forward
        if (agent_dir - 1) % 4 == best_dir:
            return 0  # turn left
        if (agent_dir + 1) % 4 == best_dir:
            return 1  # turn right
        return 0  # 180 turn — turn left
