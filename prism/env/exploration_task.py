"""Exploration task wrapper for Exp B: hidden goals discovery.

Wraps a MiniGrid env to create an exploration task with multiple
invisible goals. The agent receives reward when first visiting a
goal position. Episode ends when all goals found or max_steps reached.
"""

import gymnasium as gym
import numpy as np


def get_room_cells(state_mapper):
    """Partition accessible cells into 4 quadrants (rooms).

    Uses the grid midpoint to divide cells into TL, TR, BL, BR quadrants.
    Works for FourRooms and similar symmetric layouts.

    Returns:
        list of 4 lists, each containing state indices for one room.
    """
    h, w = state_mapper.get_grid_shape()
    mid_x = w // 2
    mid_y = h // 2

    rooms = [[], [], [], []]  # TL, TR, BL, BR
    for idx in range(state_mapper.n_states):
        x, y = state_mapper.get_pos(idx)
        if x < mid_x and y < mid_y:
            rooms[0].append(idx)
        elif x > mid_x and y < mid_y:
            rooms[1].append(idx)
        elif x < mid_x and y > mid_y:
            rooms[2].append(idx)
        elif x > mid_x and y > mid_y:
            rooms[3].append(idx)
        # Cells exactly on midline are skipped (wall/passage)

    return rooms


def place_goals(state_mapper, n_goals=4, rng=None):
    """Place n_goals randomly, one per room.

    Args:
        state_mapper: StateMapper instance.
        n_goals: Number of goals (default 4, one per room).
        rng: numpy random Generator.

    Returns:
        list of (x, y) goal positions.
    """
    if rng is None:
        rng = np.random.default_rng()

    rooms = get_room_cells(state_mapper)
    goals = []
    for i in range(min(n_goals, len(rooms))):
        if rooms[i]:
            idx = rng.choice(rooms[i])
            goals.append(state_mapper.get_pos(idx))
    return goals


class ExplorationTaskWrapper(gym.Wrapper):
    """Wraps MiniGrid for multi-goal exploration task.

    - Places invisible goals (not rendered in grid)
    - Gives +1 reward on first visit to each goal position
    - Tracks discovery times
    - Terminates when all goals found or max_steps reached
    """

    def __init__(self, env, goal_positions, max_steps=2000):
        """
        Args:
            env: MiniGrid gymnasium environment.
            goal_positions: list of (x, y) goal positions.
            max_steps: Maximum steps before forced termination.
        """
        super().__init__(env)
        self.goal_positions = [tuple(p) for p in goal_positions]
        self._max_steps = max_steps
        self._discovered = set()
        self._step_count = 0
        self._discovery_times = {}
        self._trajectory = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._discovered = set()
        self._step_count = 0
        self._discovery_times = {}
        self._trajectory = []

        # Remove MiniGrid's built-in goal to prevent early termination
        grid = self.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    grid.set(x, y, None)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        pos = tuple(self.unwrapped.agent_pos)
        self._trajectory.append(pos)

        # Check goal discovery
        if pos in self.goal_positions and pos not in self._discovered:
            self._discovered.add(pos)
            self._discovery_times[pos] = self._step_count
            reward = 1.0

        # Override termination: only when all found or max_steps
        all_found = len(self._discovered) == len(self.goal_positions)
        terminated = all_found
        truncated = self._step_count >= self._max_steps

        info["goals_found"] = len(self._discovered)
        info["goals_total"] = len(self.goal_positions)
        info["discovery_times"] = dict(self._discovery_times)
        info["all_found"] = all_found
        info["trajectory_len"] = self._step_count

        return obs, reward, terminated, truncated, info

    @property
    def trajectory(self):
        """Return the full trajectory as list of (x,y) positions."""
        return list(self._trajectory)

    @property
    def discovery_times(self):
        """Return dict mapping goal_pos -> step when discovered."""
        return dict(self._discovery_times)
