"""Meta-SR: uncertainty map U(s) and confidence signal C(s).

PRISM's core contribution. Constructs an iso-structural uncertainty map
from SR prediction errors. Same state indexing as M, same spatial granularity.

Three regimes for U(s):
    1. Unvisited: U(s) = U_prior (maximum uncertainty)
    2. Cold start (visits < K): U(s) = U_prior * decay^visits
    3. Sufficient data (visits >= K): U(s) = mean of last K scalar errors

References:
    Ambrogioni & Olafsdottir (2023), master.md section 5.4
"""

import numpy as np
from collections import deque


class MetaSR:
    """Metacognitive layer built on SR prediction errors.

    U(s): uncertainty map, iso-structural to M
    C(s): calibrated confidence signal via sigmoid
    Change detection via recent uncertainty monitoring

    Attributes:
        U: Uncertainty array of shape (n_states,).
        visit_counts: Visit count per state.
    """

    def __init__(self, n_states: int, buffer_size: int = 20,
                 U_prior: float = 0.8, decay: float = 0.85,
                 beta: float = 10.0, theta_C: float = 0.3,
                 theta_change: float = 0.5):
        """Initialize the meta-SR layer.

        Args:
            n_states: Number of states (must match SRLayer).
            buffer_size: K — circular buffer size per state.
            U_prior: Prior uncertainty for unvisited states.
            decay: Per-visit decay rate for cold-start regime.
            beta: Sigmoid steepness for confidence signal.
            theta_C: Sigmoid center for confidence threshold.
            theta_change: Change detection threshold.
        """
        self.n_states = n_states
        self.buffer_size = buffer_size
        self.U_prior = U_prior
        self.decay = decay
        self.beta = beta
        self.theta_C = theta_C
        self.theta_change = theta_change

        # Per-state circular buffers of scalar prediction errors
        self._buffers = [deque(maxlen=buffer_size) for _ in range(n_states)]
        self.visit_counts = np.zeros(n_states, dtype=np.int64)

        # Running normalization for delta scalars
        self._all_deltas = deque(maxlen=5000)
        self._p99_cache = 0.0
        self._observe_count = 0

        # Recent visit tracking for change detection
        self._recent_visits = deque(maxlen=50)

    def observe(self, s: int, delta_M: np.ndarray):
        """Process a new SR prediction error at state s.

        Computes the scalar compression ||delta_M||_2, normalizes it,
        and stores in the per-state circular buffer.

        Args:
            s: State index where the transition occurred.
            delta_M: Full TD error vector from SRLayer.update().
        """
        # Scalar compression: L2 norm
        delta_scalar = float(np.linalg.norm(delta_M))

        # Store raw delta for running normalization
        self._all_deltas.append(delta_scalar)
        self._observe_count += 1

        # Normalize to [0, 1] via adaptive percentile clipping
        if len(self._all_deltas) >= 10:
            # Recompute p99 every 100 steps (expensive on large deque)
            if self._observe_count % 100 == 0 or self._p99_cache == 0.0:
                self._p99_cache = float(np.percentile(list(self._all_deltas), 99))
            p99 = self._p99_cache
            if p99 > 0:
                delta_normalized = min(delta_scalar / p99, 1.0)
            else:
                delta_normalized = 0.0
        else:
            # Not enough data yet — use raw value clipped to [0, 1]
            delta_normalized = min(delta_scalar, 1.0)

        self._buffers[s].append(delta_normalized)
        self.visit_counts[s] += 1
        self._recent_visits.append(s)

    def uncertainty(self, s: int) -> float:
        """Compute U(s) — uncertainty at state s.

        Three regimes:
            visits == 0:       U = U_prior
            0 < visits < K:    U = U_prior * decay^visits
            visits >= K:       U = mean(buffer)

        Returns:
            Uncertainty value in [0, 1].
        """
        visits = self.visit_counts[s]

        if visits == 0:
            return self.U_prior
        elif visits < self.buffer_size:
            return self.U_prior * (self.decay ** visits)
        else:
            return float(np.mean(self._buffers[s]))

    def all_uncertainties(self) -> np.ndarray:
        """Compute U for all states. Returns array of shape (n_states,)."""
        return np.array([self.uncertainty(s) for s in range(self.n_states)])

    def confidence(self, s: int) -> float:
        """Confidence signal C(s) in [0, 1]. 1 = high confidence.

        C(s) = 1 / (1 + exp(beta * (U(s) - theta_C)))
        High U -> low C, low U -> high C.
        """
        U_s = self.uncertainty(s)
        return 1.0 / (1.0 + np.exp(self.beta * (U_s - self.theta_C)))

    def all_confidences(self) -> np.ndarray:
        """Compute C for all states. Returns array of shape (n_states,)."""
        U = self.all_uncertainties()
        return 1.0 / (1.0 + np.exp(self.beta * (U - self.theta_C)))

    def detect_change(self) -> bool:
        """Detect structural change from recent uncertainty spike.

        Computes mean U over recently visited states.
        Returns True if change_score > theta_change.
        """
        if len(self._recent_visits) == 0:
            return False

        recent_unique = set(self._recent_visits)
        change_score = np.mean([self.uncertainty(s) for s in recent_unique])
        return float(change_score) > self.theta_change
