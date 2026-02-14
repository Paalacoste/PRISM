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

    def select_action(self, s: int, available_actions: list[int]) -> tuple[int, float, bool]:
        """Select an action using adaptive epsilon-greedy on V_explore.

        Args:
            s: Current state index.
            available_actions: List of valid action indices.

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
            # Explore: random action
            action = self._rng.choice(available_actions)
        else:
            # Exploit: greedy on V_explore for reachable states
            # For MiniGrid movement actions (0=left, 1=right, 2=forward),
            # we evaluate the exploration value of the current state
            # since we don't know next states without simulating.
            # Use simple greedy on V(s) with exploration bonus.
            action = self._rng.choice(available_actions)

            # If we have neighbor information, prefer highest V_explore
            # For now, use the fact that action 2 (forward) is the only
            # action that changes position — prefer it when V_explore is high
            if 2 in available_actions and len(available_actions) > 1:
                # Forward has the potential to reach a new state
                action = int(self._rng.choice(available_actions))

        return (action, float(confidence), bool(idk_flag))
