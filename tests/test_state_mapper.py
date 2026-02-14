"""Tests for StateMapper."""

import numpy as np
import pytest

from prism.env.state_mapper import StateMapper


class TestStateMapper:

    def test_n_states_positive(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        assert mapper.n_states > 0

    def test_n_states_matches_accessible_cells(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        # Count accessible cells manually
        grid = four_rooms_env.unwrapped.grid
        accessible = 0
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is None or (cell is not None and cell.type in ("door", "goal")):
                    accessible += 1
        assert mapper.n_states == accessible

    def test_pos_to_idx_and_back(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        for idx in range(mapper.n_states):
            pos = mapper.get_pos(idx)
            assert mapper.get_index(pos) == idx

    def test_all_indices_contiguous(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        indices = sorted(mapper.idx_to_pos.keys())
        assert indices == list(range(mapper.n_states))

    def test_agent_pos_is_mapped(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        agent_pos = tuple(four_rooms_env.unwrapped.agent_pos)
        idx = mapper.get_index(agent_pos)
        assert 0 <= idx < mapper.n_states

    def test_wall_pos_raises(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        # (0, 0) is always a wall in MiniGrid (outer boundary)
        with pytest.raises(KeyError):
            mapper.get_index((0, 0))

    def test_grid_shape(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        h, w = mapper.get_grid_shape()
        assert h == 19
        assert w == 19

    def test_to_grid_shape(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        values = np.arange(mapper.n_states, dtype=float)
        grid = mapper.to_grid(values)
        assert grid.shape == (19, 19)

    def test_to_grid_walls_are_nan(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        values = np.ones(mapper.n_states)
        grid = mapper.to_grid(values)
        # (0, 0) is a wall -> should be NaN
        assert np.isnan(grid[0, 0])

    def test_to_grid_accessible_filled(self, four_rooms_env):
        mapper = StateMapper(four_rooms_env)
        values = np.ones(mapper.n_states) * 42.0
        grid = mapper.to_grid(values)
        # Agent position should have the value
        ax, ay = four_rooms_env.unwrapped.agent_pos
        assert grid[ay, ax] == 42.0

    def test_deterministic_mapping(self):
        """Same seed produces same mapping."""
        import gymnasium as gym
        import minigrid  # noqa: F401

        env1 = gym.make("MiniGrid-FourRooms-v0")
        env1.reset(seed=42)
        m1 = StateMapper(env1)
        env1.close()

        env2 = gym.make("MiniGrid-FourRooms-v0")
        env2.reset(seed=42)
        m2 = StateMapper(env2)
        env2.close()

        assert m1.n_states == m2.n_states
        assert m1.pos_to_idx == m2.pos_to_idx
