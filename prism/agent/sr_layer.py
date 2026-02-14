"""Tabular Successor Representation learned via TD(0).

The key interface requirement: update() must return the full TD error vector
delta_M, which feeds into the meta-SR layer.

References:
    Dayan (1993), Stachenfeld et al. (2017), master.md section 5.3
"""

import numpy as np


class SRLayer:
    """Tabular successor representation with TD(0) learning.

    M(s, s') = E[sum_t gamma^t * I(s_t = s') | s_0 = s, pi]
    V(s) = M(s, :) . R

    Attributes:
        M: SR matrix of shape (n_states, n_states). Initialized to identity.
        R: Reward vector of shape (n_states,). Initialized to zeros.
    """

    def __init__(self, n_states: int, gamma: float = 0.95,
                 alpha_M: float = 0.1, alpha_R: float = 0.3):
        """Initialize the SR layer.

        Args:
            n_states: Number of states (N). Determines M shape (N x N).
            gamma: Discount factor / temporal horizon.
            alpha_M: Learning rate for M matrix (slower, structural).
            alpha_R: Learning rate for reward vector R (faster, local).
        """
        self.n_states = n_states
        self.gamma = gamma
        self.alpha_M = alpha_M
        self.alpha_R = alpha_R

        # M initialized to identity: each state predicts itself
        self.M = np.eye(n_states, dtype=np.float64)
        # R initialized to zero: no prior reward knowledge
        self.R = np.zeros(n_states, dtype=np.float64)

    def update(self, s: int, s_next: int, reward: float) -> np.ndarray:
        """Perform one TD(0) update step.

        Updates M(s, :) and R(s_next), returns the TD error vector delta_M
        which is passed to the meta-SR layer.

        Args:
            s: Current state index.
            s_next: Next state index after transition.
            reward: Reward received on this transition.

        Returns:
            delta_M: TD error vector of shape (n_states,).
                     delta_M = e(s') + gamma * M(s', :) - M(s, :)
        """
        # One-hot for next state
        e_next = np.zeros(self.n_states, dtype=np.float64)
        e_next[s_next] = 1.0

        # TD error on M row
        delta_M = e_next + self.gamma * self.M[s_next] - self.M[s]

        # Update M
        self.M[s] += self.alpha_M * delta_M

        # Update R (decoupled, faster learning)
        self.R[s_next] += self.alpha_R * (reward - self.R[s_next])

        return delta_M

    def value(self, s: int) -> float:
        """Compute V(s) = M[s, :] . R."""
        return float(self.M[s] @ self.R)

    def all_values(self) -> np.ndarray:
        """Compute V for all states: M @ R."""
        return self.M @ self.R
