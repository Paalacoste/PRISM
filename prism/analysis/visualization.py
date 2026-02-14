"""Visualization: SR heatmaps, uncertainty maps, grid overlays.

References:
    master.md section 6 (figures)
"""

import numpy as np


def plot_sr_heatmap(M, state_mapper, source_state=None, ax=None):
    """Plot SR matrix row as a heatmap on the grid.

    Args:
        M: SR matrix of shape (n_states, n_states).
        state_mapper: A StateMapper instance.
        source_state: State index whose SR row to display.
            If None, plots the mean across all rows.
        ax: Optional matplotlib axes.

    Returns:
        fig: The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    if source_state is not None:
        values = M[source_state]
        title = f"SR from state {source_state} {state_mapper.get_pos(source_state)}"
    else:
        values = M.mean(axis=0)
        title = "SR mean occupancy"

    grid = state_mapper.to_grid(values)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(grid, cmap="hot", interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_value_map(V, state_mapper, ax=None):
    """Plot value function V(s) as a heatmap on the grid.

    Args:
        V: Value vector of shape (n_states,).
        state_mapper: A StateMapper instance.
        ax: Optional matplotlib axes.

    Returns:
        fig: The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    grid = state_mapper.to_grid(V)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title("Value function V(s)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_uncertainty_map(U, state_mapper, ax=None):
    """Plot uncertainty U(s) as a heatmap on the grid.

    Args:
        U: Uncertainty vector of shape (n_states,).
        state_mapper: A StateMapper instance.
        ax: Optional matplotlib axes.

    Returns:
        fig: The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    grid = state_mapper.to_grid(U)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Uncertainty U(s)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
