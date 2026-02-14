"""Spectral decomposition of SR matrix for grid cell validation.

Eigenvectors of M should show room-level structure, reproducing
Stachenfeld et al. (2017) Figure 3.

References:
    Stachenfeld et al. (2017), master.md section 7.3
"""

import numpy as np
from scipy import linalg


def sr_eigenvectors(M, k=6):
    """Extract top-k eigenvectors of SR matrix M.

    Args:
        M: SR matrix of shape (n_states, n_states).
        k: Number of top eigenvectors to extract.

    Returns:
        eigenvalues: Array of shape (k,), sorted descending.
        eigenvectors: Array of shape (n_states, k), columns are eigenvectors.
    """
    n = M.shape[0]
    k = min(k, n)

    # Use eigh for symmetric part; SR matrices are approximately symmetric
    # for a converged random walk policy
    M_sym = (M + M.T) / 2
    eigenvalues, eigenvectors = linalg.eigh(
        M_sym, subset_by_index=[n - k, n - 1]
    )

    # Sort descending by eigenvalue
    idx = np.argsort(-eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def plot_eigenvectors(eigenvectors, eigenvalues, state_mapper, k=6, ax=None):
    """Plot eigenvectors as heatmaps on the grid.

    Args:
        eigenvectors: Array of shape (n_states, k).
        eigenvalues: Array of shape (k,).
        state_mapper: A StateMapper instance.
        k: Number of eigenvectors to plot.
        ax: Optional matplotlib axes array. If None, creates a new figure.

    Returns:
        fig: The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    k = min(k, eigenvectors.shape[1])
    ncols = 3
    nrows = (k + ncols - 1) // ncols

    if ax is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    else:
        fig = ax.flat[0].get_figure() if hasattr(ax, 'flat') else ax.get_figure()
        axes = ax

    axes_flat = np.array(axes).flat

    for i in range(k):
        grid = state_mapper.to_grid(eigenvectors[:, i])
        im = axes_flat[i].imshow(grid, cmap="RdBu_r", interpolation="nearest")
        axes_flat[i].set_title(f"EV {i + 1}, λ={eigenvalues[i]:.3f}")
        plt.colorbar(im, ax=axes_flat[i], fraction=0.046)

    # Hide unused axes
    for i in range(k, len(list(axes_flat))):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    return fig
