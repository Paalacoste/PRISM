"""Simplified grid world for pedagogical demonstrations.

No MiniGrid dependency — pure numpy. Supports instant SR computation
for interactive ipywidgets demos.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class ToyGrid:
    """Simple N×N grid world with walls and a goal.

    Parameters
    ----------
    size : int or tuple
        Grid size. Int for square grid, (rows, cols) for rectangular.
    walls : set of (row, col) tuples
        Positions occupied by walls (inaccessible).
    goal : (row, col) or None
        Goal position. If None, no goal is set.
    """

    def __init__(self, size=5, walls=None, goal=None):
        if isinstance(size, int):
            self.rows, self.cols = size, size
        else:
            self.rows, self.cols = size

        self.walls = set(walls) if walls else set()
        self.goal = goal

        # Enumerate accessible states
        self.states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:
                    self.states.append((r, c))

        self.n_states = len(self.states)
        self.pos_to_idx = {pos: i for i, pos in enumerate(self.states)}
        self.idx_to_pos = {i: pos for i, pos in enumerate(self.states)}

        # Actions: up, down, left, right
        self._actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _step(self, pos, action_idx):
        """Take one step from pos. Returns new pos (stays if wall/boundary)."""
        dr, dc = self._actions[action_idx]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
            return (nr, nc)
        return pos  # bounce back

    def transition_matrix(self, policy='uniform'):
        """Compute one-step transition matrix T.

        Parameters
        ----------
        policy : str
            'uniform' for equal probability over 4 actions.

        Returns
        -------
        T : ndarray (n_states, n_states)
            T[i, j] = P(s' = j | s = i)
        """
        n = self.n_states
        T = np.zeros((n, n))

        for i, pos in enumerate(self.states):
            for a in range(4):
                next_pos = self._step(pos, a)
                j = self.pos_to_idx[next_pos]
                T[i, j] += 0.25  # uniform over 4 actions

        return T

    def true_sr(self, gamma=0.95):
        """Compute analytical SR matrix M* = (I - γT)^{-1}.

        Parameters
        ----------
        gamma : float
            Discount factor.

        Returns
        -------
        M_star : ndarray (n_states, n_states)
        """
        T = self.transition_matrix()
        I = np.eye(self.n_states)
        return np.linalg.inv(I - gamma * T)

    def reward_vector(self, goal=None):
        """Create reward vector R with 1.0 at goal, 0 elsewhere.

        Parameters
        ----------
        goal : (row, col) or None
            Goal position. Uses self.goal if None.

        Returns
        -------
        R : ndarray (n_states,)
        """
        g = goal if goal is not None else self.goal
        R = np.zeros(self.n_states)
        if g is not None and g in self.pos_to_idx:
            R[self.pos_to_idx[g]] = 1.0
        return R

    def td_update(self, M, s_idx, s_next_idx, gamma=0.95, alpha=0.1):
        """Perform one TD(0) update on M.

        Parameters
        ----------
        M : ndarray (n_states, n_states)
            Current SR estimate.
        s_idx, s_next_idx : int
            Current and next state indices.
        gamma, alpha : float
            Discount and learning rate.

        Returns
        -------
        delta : ndarray (n_states,)
            TD error vector.
        M_new : ndarray (n_states, n_states)
            Updated SR matrix (copy).
        """
        e = np.zeros(self.n_states)
        e[s_next_idx] = 1.0
        delta = e + gamma * M[s_next_idx] - M[s_idx]
        M_new = M.copy()
        M_new[s_idx] += alpha * delta
        return delta, M_new

    def random_walk(self, n_steps, start=None, seed=None):
        """Generate a random walk trajectory.

        Parameters
        ----------
        n_steps : int
            Number of steps.
        start : int or None
            Starting state index. Random if None.
        seed : int or None
            Random seed.

        Returns
        -------
        trajectory : list of int
            State indices visited.
        """
        rng = np.random.RandomState(seed)
        if start is None:
            s = rng.randint(self.n_states)
        else:
            s = start

        trajectory = [s]
        for _ in range(n_steps):
            a = rng.randint(4)
            next_pos = self._step(self.idx_to_pos[s], a)
            s = self.pos_to_idx[next_pos]
            trajectory.append(s)
        return trajectory

    def to_grid(self, values, fill=np.nan):
        """Map per-state vector to 2D grid for visualization.

        Parameters
        ----------
        values : ndarray (n_states,)
            One value per accessible state.
        fill : float
            Value for wall cells.

        Returns
        -------
        grid : ndarray (rows, cols)
        """
        grid = np.full((self.rows, self.cols), fill)
        for i, pos in enumerate(self.states):
            grid[pos[0], pos[1]] = values[i]
        return grid

    def plot(self, values=None, ax=None, title=None, cmap='viridis',
             vmin=None, vmax=None, show_walls=True, show_goal=True):
        """Visualize the grid with optional values overlay.

        Parameters
        ----------
        values : ndarray (n_states,) or None
            Values to display. If None, shows grid structure only.
        ax : matplotlib Axes or None
        title : str or None
        cmap : str
        vmin, vmax : float or None
        show_walls : bool
        show_goal : bool

        Returns
        -------
        ax : matplotlib Axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        if values is not None:
            grid = self.to_grid(values, fill=np.nan)
            im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                          origin='upper', interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Show grid structure: white = accessible, gray = wall
            grid = np.ones((self.rows, self.cols))
            for w in self.walls:
                grid[w[0], w[1]] = 0
            cmap_struct = ListedColormap(['#404040', '#f0f0f0'])
            ax.imshow(grid, cmap=cmap_struct, origin='upper',
                     interpolation='nearest')

        # Mark walls
        if show_walls and values is not None:
            for w in self.walls:
                ax.add_patch(plt.Rectangle((w[1]-0.5, w[0]-0.5), 1, 1,
                            fill=True, color='gray', alpha=0.8))

        # Mark goal
        if show_goal and self.goal is not None:
            ax.plot(self.goal[1], self.goal[0], 'r*', markersize=15, zorder=5)

        # Grid lines
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='both', bottom=False, left=False,
                      labelbottom=False, labelleft=False)

        if title:
            ax.set_title(title)
        return ax

    # --- Predefined configurations ---

    @classmethod
    def two_rooms(cls):
        """5×5 grid with central wall and one passage.

        Layout (. = open, # = wall):
            . . # . .
            . . # . .
            . . . . .    <- passage at (2, 2)
            . . # . .
            . . # . .
        """
        walls = {(0, 2), (1, 2), (3, 2), (4, 2)}
        return cls(size=5, walls=walls, goal=(0, 4))

    @classmethod
    def open_field(cls):
        """5×5 grid with no walls (25 states)."""
        return cls(size=5, walls=set(), goal=(4, 4))

    @classmethod
    def corridor(cls):
        """7×1 linear corridor (7 states)."""
        return cls(size=(1, 7), walls=set(), goal=(0, 6))
