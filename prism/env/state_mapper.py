"""Bidirectional mapping between MiniGrid grid positions and SR matrix indices.

Converts (x, y) positions from MiniGrid into integer indices for the
N x N successor representation matrix M. Excludes walls, includes doors.

References:
    master.md section 7.2
"""

import numpy as np


class StateMapper:
    """Maps MiniGrid positions to SR state indices and back.

    After construction, provides:
        pos_to_idx: dict mapping (x, y) -> int
        idx_to_pos: dict mapping int -> (x, y)
        n_states: total number of accessible states

    Direction is ignored (state = position only), following
    Stachenfeld et al. (2017).
    """

    def __init__(self, env):
        """Build the state map from a MiniGrid environment.

        Args:
            env: A MiniGrid environment (or its unwrapped base).
        """
        self.pos_to_idx = {}
        self.idx_to_pos = {}
        self.n_states = 0

        grid = env.unwrapped.grid
        self._grid_width = grid.width
        self._grid_height = grid.height

        idx = 0
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get(x, y)
                if cell is None or (cell is not None and cell.type in ("door", "goal")):
                    self.pos_to_idx[(x, y)] = idx
                    self.idx_to_pos[idx] = (x, y)
                    idx += 1

        self.n_states = idx

    def get_index(self, pos: tuple[int, int]) -> int:
        """Convert grid position to state index.

        Args:
            pos: (x, y) grid position.

        Returns:
            Integer state index.

        Raises:
            KeyError: If position is not accessible (wall).
        """
        return self.pos_to_idx[tuple(pos)]

    def get_pos(self, idx: int) -> tuple[int, int]:
        """Convert state index to grid position.

        Args:
            idx: Integer state index.

        Returns:
            (x, y) grid position.

        Raises:
            KeyError: If index is out of range.
        """
        return self.idx_to_pos[idx]

    def get_grid_shape(self) -> tuple[int, int]:
        """Return the (height, width) of the full grid."""
        return (self._grid_height, self._grid_width)

    def to_grid(self, values: np.ndarray, fill_value=np.nan) -> np.ndarray:
        """Map a per-state vector onto a 2D grid for visualization.

        Args:
            values: Array of shape (n_states,).
            fill_value: Value for inaccessible cells (walls).

        Returns:
            2D array of shape (height, width).
        """
        grid = np.full(self.get_grid_shape(), fill_value)
        for idx in range(self.n_states):
            x, y = self.idx_to_pos[idx]
            grid[y, x] = values[idx]
        return grid
