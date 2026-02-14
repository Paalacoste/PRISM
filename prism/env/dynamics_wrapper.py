"""Gymnasium wrapper adding controlled perturbations to MiniGrid environments.

Supports reward shifts, door blocking/opening, and scheduled perturbations.
MiniGrid rebuilds the grid on every reset(), so perturbations must be
re-applied after each reset.

References:
    master.md section 5.2
"""

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal


class DynamicsWrapper(gym.Wrapper):
    """Adds controlled dynamic perturbations to a MiniGrid environment.

    Perturbation types:
        'reward_shift': Move the goal to a new position
        'door_block': Block a passage with a wall
        'door_open': Remove a wall to open a new passage
    """

    def __init__(self, env, seed=None):
        super().__init__(env)
        self._active_perturbations = []
        self._rng = np.random.default_rng(seed)

    def apply_perturbation(self, ptype: str, **kwargs):
        """Apply a perturbation and record it for re-application on reset.

        Args:
            ptype: One of 'reward_shift', 'door_block', 'door_open'.
            **kwargs: Perturbation-specific arguments.
                reward_shift: new_goal_pos=(x, y)
                door_block: wall_pos=(x, y)
                door_open: wall_pos=(x, y)
        """
        perturbation = {"ptype": ptype, **kwargs}
        self._active_perturbations.append(perturbation)
        self._apply_single(perturbation)

    def clear_perturbations(self):
        """Remove all active perturbations."""
        self._active_perturbations.clear()

    def reset(self, **kwargs):
        """Reset environment and re-apply all active perturbations."""
        obs, info = self.env.reset(**kwargs)
        for p in self._active_perturbations:
            self._apply_single(p)
        return obs, info

    def _apply_single(self, perturbation):
        """Apply a single perturbation to the current grid."""
        grid = self.unwrapped.grid
        ptype = perturbation["ptype"]

        if ptype == "reward_shift":
            new_pos = perturbation["new_goal_pos"]
            # Remove existing goal(s)
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type == "goal":
                        grid.set(x, y, None)
            # Place new goal
            grid.set(new_pos[0], new_pos[1], Goal())

        elif ptype == "door_block":
            pos = perturbation["wall_pos"]
            grid.set(pos[0], pos[1], Wall())

        elif ptype == "door_open":
            pos = perturbation["wall_pos"]
            grid.set(pos[0], pos[1], None)

        else:
            raise ValueError(f"Unknown perturbation type: {ptype}")

    def get_true_transition_matrix(self, state_mapper) -> np.ndarray:
        """Compute ground-truth one-step transition matrix under uniform random policy.

        For each state s and each movement action (left, right, forward),
        determines the resulting state s'. Under a uniform random policy over
        the 3 movement actions, T[s, s'] = count / 3.

        This is the one-step transition matrix, NOT the full SR matrix M*.
        The true SR is M* = (I - gamma * T)^{-1}.

        Args:
            state_mapper: A StateMapper instance for this environment.

        Returns:
            T: Transition matrix of shape (n_states, n_states).
        """
        n = state_mapper.n_states
        T = np.zeros((n, n), dtype=np.float64)

        env = self.unwrapped
        # Save current state
        saved_pos = tuple(env.agent_pos)
        saved_dir = env.agent_dir

        movement_actions = [0, 1, 2]  # turn_left, turn_right, forward

        for s_idx in range(n):
            pos = state_mapper.get_pos(s_idx)
            for direction in range(4):
                # Place agent at (pos, direction)
                env.agent_pos = np.array(pos)
                env.agent_dir = direction

                for action in movement_actions:
                    # Simulate the action
                    env.agent_pos = np.array(pos)
                    env.agent_dir = direction

                    if action == 0:  # turn left
                        new_dir = (direction - 1) % 4
                        new_pos = pos
                    elif action == 1:  # turn right
                        new_dir = (direction + 1) % 4
                        new_pos = pos
                    else:  # forward
                        # Compute forward position based on direction
                        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
                        fx, fy = pos[0] + dx, pos[1] + dy
                        # Check if forward cell is walkable
                        fwd_cell = env.grid.get(fx, fy)
                        if fwd_cell is None or (fwd_cell is not None and fwd_cell.can_overlap()):
                            new_pos = (fx, fy)
                        else:
                            new_pos = pos  # blocked, stay

                    # Map new_pos to index
                    try:
                        s_next_idx = state_mapper.get_index(new_pos)
                    except KeyError:
                        s_next_idx = s_idx  # stay if somehow outside map

                    # Under uniform direction assumption: weight = 1/(4*3)
                    T[s_idx, s_next_idx] += 1.0 / (4 * 3)

        # Restore agent state
        env.agent_pos = saved_pos
        env.agent_dir = saved_dir

        return T
