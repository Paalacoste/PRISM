"""SR-Count-Matched baseline: count-based bonus calibrated to match PRISM's U(s) decay.

The bonus uses a lookup table u_profile[visits(s)] that reproduces the average
temporal decay profile of PRISM's uncertainty signal U(s). This isolates the
contribution of the SR's structural information from the temporal decay shape.

If PRISM beats Count-Matched, the advantage comes from the *content* of U(s)
(local transition structure), not merely its decay profile.

References:
    exp_b_improvements.md — Amélioration 2
"""

from collections import defaultdict

import numpy as np

from prism.baselines.base_agent import BaseSRAgent
from prism.agent.sr_layer import SRLayer
from prism.agent.meta_sr import MetaSR
from prism.config import PRISMConfig


class SRCountMatched(BaseSRAgent):
    """SR agent with exploration bonus matching PRISM's U(s) decay profile.

    bonus(s) = u_profile[min(visits(s), len(u_profile)-1)]

    The profile is calibrated empirically so that the average bonus at a given
    visit count equals the average U(s) observed in PRISM at the same count.
    """

    def __init__(self, n_states, state_mapper, u_profile, epsilon=0.1,
                 **kwargs):
        """
        Args:
            u_profile: 1D array where u_profile[n] = E[U(s) | visits(s) = n].
                       Obtained from calibrate_u_profile().
            epsilon: Fixed exploration rate.
        """
        super().__init__(n_states, state_mapper, **kwargs)
        self._u_profile = np.asarray(u_profile, dtype=np.float64)
        self._epsilon = epsilon

    def get_epsilon(self, s, total_steps):
        return self._epsilon

    def exploration_bonus(self, s):
        n = min(self.visit_counts[s], len(self._u_profile) - 1)
        return float(self._u_profile[n])


def calibrate_u_profile(env, state_mapper, n_episodes=50, n_runs=20,
                        config=None, env_seed=42, max_steps=2000):
    """Calibrate the U(s) decay profile by running PRISM under random policy.

    Collects (visit_count, U(s)) pairs ONLINE at each step (before update),
    then averages U by visit count to produce the profile.

    Args:
        env: MiniGrid gymnasium environment.
        state_mapper: StateMapper instance.
        n_episodes: Episodes per calibration run.
        n_runs: Number of independent calibration runs.
        config: PRISMConfig (uses defaults if None).
        env_seed: Fixed seed for env.reset() (preserves grid structure).

    Returns:
        np.ndarray of shape (max_visits+1,) where profile[n] = E[U(s)|visits=n].
    """
    if config is None:
        config = PRISMConfig()

    n_states = state_mapper.n_states
    u_by_visits = defaultdict(list)

    for run in range(n_runs):
        # Fresh PRISM components per run
        sr = SRLayer(
            n_states, config.sr.gamma, config.sr.alpha_M, config.sr.alpha_R,
        )
        meta_sr = MetaSR(
            n_states,
            buffer_size=config.meta_sr.buffer_size,
            U_prior=config.meta_sr.U_prior,
            decay=config.meta_sr.decay,
            beta=config.meta_sr.beta,
            theta_C=config.meta_sr.theta_C,
            theta_change=config.meta_sr.theta_change,
        )
        rng = np.random.default_rng(env_seed + 1000 + run)

        for _ep in range(n_episodes):
            obs, _ = env.reset(seed=env_seed)
            env.unwrapped.max_steps = max_steps + 100
            s = state_mapper.get_index(tuple(env.unwrapped.agent_pos))

            done = False
            step = 0
            while not done and step < max_steps:
                # ONLINE: record (visits, U) BEFORE update
                visits_s = int(meta_sr.visit_counts[s])
                u_s = meta_sr.uncertainty(s)
                u_by_visits[visits_s].append(u_s)

                # Random action
                action = int(rng.choice([0, 1, 2]))
                obs, _r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                s_next = state_mapper.get_index(
                    tuple(env.unwrapped.agent_pos)
                )
                # Update SR + Meta-SR
                delta_M = sr.update(s, s_next, 0.0)
                meta_sr.observe(s, delta_M)

                s = s_next
                step += 1

    # Build profile array
    if not u_by_visits:
        return np.array([config.meta_sr.U_prior])

    max_visits = max(u_by_visits.keys())
    profile = np.zeros(max_visits + 1)

    for n in range(max_visits + 1):
        if n in u_by_visits and len(u_by_visits[n]) >= 3:
            profile[n] = np.mean(u_by_visits[n])
        elif n == 0:
            profile[n] = config.meta_sr.U_prior
        else:
            # Interpolate from previous bin
            profile[n] = profile[n - 1]

    return profile
