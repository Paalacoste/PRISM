"""Tests for DynamicsWrapper."""

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import pytest

from prism.env.dynamics_wrapper import DynamicsWrapper
from prism.env.state_mapper import StateMapper


@pytest.fixture
def wrapped_env():
    """Create a DynamicsWrapper around FourRooms."""
    env = gym.make("MiniGrid-FourRooms-v0")
    wrapped = DynamicsWrapper(env, seed=42)
    wrapped.reset(seed=42)
    yield wrapped
    wrapped.close()


@pytest.fixture
def mapper(wrapped_env):
    return StateMapper(wrapped_env)


class TestDynamicsWrapperBasic:

    def test_wraps_env(self, wrapped_env):
        assert isinstance(wrapped_env, DynamicsWrapper)
        obs, info = wrapped_env.reset(seed=42)
        assert obs is not None

    def test_step_works(self, wrapped_env):
        obs, reward, terminated, truncated, info = wrapped_env.step(2)
        assert obs is not None


class TestRewardShift:

    def test_goal_moves(self, wrapped_env):
        # Find an empty cell for the new goal
        grid = wrapped_env.unwrapped.grid
        new_pos = None
        for x in range(1, grid.width - 1):
            for y in range(1, grid.height - 1):
                cell = grid.get(x, y)
                if cell is None:
                    new_pos = (x, y)
                    break
            if new_pos:
                break

        wrapped_env.apply_perturbation("reward_shift", new_goal_pos=new_pos)

        # Verify new goal position
        cell = grid.get(new_pos[0], new_pos[1])
        assert cell is not None
        assert cell.type == "goal"

    def test_old_goal_removed(self, wrapped_env):
        grid = wrapped_env.unwrapped.grid
        # Find current goal
        old_goal_pos = None
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    old_goal_pos = (x, y)

        # Shift to a different position
        new_pos = (1, 1)
        if new_pos == old_goal_pos:
            new_pos = (2, 1)
        wrapped_env.apply_perturbation("reward_shift", new_goal_pos=new_pos)

        # Old position should no longer have goal
        old_cell = grid.get(old_goal_pos[0], old_goal_pos[1])
        assert old_cell is None or old_cell.type != "goal"


class TestDoorBlock:

    def test_wall_placed(self, wrapped_env):
        grid = wrapped_env.unwrapped.grid
        # Find a passage (empty cell adjacent to a wall)
        pos = (7, 9)  # Known passage in FourRooms
        if grid.get(pos[0], pos[1]) is None:
            wrapped_env.apply_perturbation("door_block", wall_pos=pos)
            cell = grid.get(pos[0], pos[1])
            assert cell is not None
            assert cell.type == "wall"


class TestDoorOpen:

    def test_wall_removed(self, wrapped_env):
        grid = wrapped_env.unwrapped.grid
        # Find an interior wall
        wall_pos = None
        for x in range(2, grid.width - 2):
            for y in range(2, grid.height - 2):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "wall":
                    wall_pos = (x, y)
                    break
            if wall_pos:
                break

        if wall_pos:
            wrapped_env.apply_perturbation("door_open", wall_pos=wall_pos)
            cell = grid.get(wall_pos[0], wall_pos[1])
            assert cell is None


class TestPerturbationPersistence:

    def test_perturbation_survives_reset(self, wrapped_env):
        new_pos = (1, 1)
        wrapped_env.apply_perturbation("reward_shift", new_goal_pos=new_pos)

        # Reset should re-apply perturbations
        wrapped_env.reset(seed=99)
        grid = wrapped_env.unwrapped.grid
        cell = grid.get(new_pos[0], new_pos[1])
        assert cell is not None
        assert cell.type == "goal"

    def test_clear_perturbations(self, wrapped_env):
        wrapped_env.apply_perturbation("reward_shift", new_goal_pos=(1, 1))
        wrapped_env.clear_perturbations()
        assert len(wrapped_env._active_perturbations) == 0


class TestTransitionMatrix:

    def test_shape(self, wrapped_env, mapper):
        T = wrapped_env.get_true_transition_matrix(mapper)
        assert T.shape == (mapper.n_states, mapper.n_states)

    def test_rows_sum_to_one(self, wrapped_env, mapper):
        T = wrapped_env.get_true_transition_matrix(mapper)
        row_sums = T.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_non_negative(self, wrapped_env, mapper):
        T = wrapped_env.get_true_transition_matrix(mapper)
        assert np.all(T >= 0)

    def test_self_transitions_exist(self, wrapped_env, mapper):
        """Turn actions don't change position, so diagonal should be > 0."""
        T = wrapped_env.get_true_transition_matrix(mapper)
        # Most states should have self-transition > 0 (turns don't move)
        assert np.sum(np.diag(T) > 0) > mapper.n_states * 0.5
