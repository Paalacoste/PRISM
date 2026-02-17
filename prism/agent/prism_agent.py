"""Full PRISM agent assembling SR + Meta-SR + Controller.

Handles the train loop, logging, and metric collection.

References:
    master.md section 5
"""

import numpy as np
from tqdm import trange

from prism.agent.sr_layer import SRLayer
from prism.agent.meta_sr import MetaSR
from prism.agent.controller import PRISMController
from prism.env.state_mapper import StateMapper
from prism.config import PRISMConfig


# Movement actions only (ignore pickup, drop, toggle, done)
MOVEMENT_ACTIONS = [0, 1, 2]  # turn_left, turn_right, forward


class PRISMAgent:
    """Complete PRISM agent for MiniGrid environments.

    Assembles:
        - StateMapper: (x,y) <-> state index
        - SRLayer: tabular SR with TD(0)
        - MetaSR: uncertainty map + confidence signal
        - PRISMController: adaptive exploration

    Attributes:
        mapper: StateMapper instance.
        sr: SRLayer instance.
        meta_sr: MetaSR instance.
        controller: PRISMController instance.
        history: List of per-episode metric dicts.
    """

    def __init__(self, env, config: PRISMConfig | None = None, seed: int = 42):
        """Initialize the PRISM agent.

        Args:
            env: A MiniGrid gymnasium environment (possibly wrapped).
            config: PRISMConfig with all hyperparameters. Uses defaults if None.
            seed: Random seed.
        """
        if config is None:
            config = PRISMConfig()

        self.env = env
        self.config = config
        self.seed = seed

        # Build state mapper from current grid
        self.mapper = StateMapper(env)

        # Build SR layer
        self.sr = SRLayer(
            n_states=self.mapper.n_states,
            gamma=config.sr.gamma,
            alpha_M=config.sr.alpha_M,
            alpha_R=config.sr.alpha_R,
        )

        # Build meta-SR layer
        self.meta_sr = MetaSR(
            n_states=self.mapper.n_states,
            buffer_size=config.meta_sr.buffer_size,
            U_prior=config.meta_sr.U_prior,
            decay=config.meta_sr.decay,
            beta=config.meta_sr.beta,
            theta_C=config.meta_sr.theta_C,
            theta_change=config.meta_sr.theta_change,
        )

        # Build controller
        self.controller = PRISMController(
            sr_layer=self.sr,
            meta_sr=self.meta_sr,
            state_mapper=self.mapper,
            epsilon_min=config.controller.epsilon_min,
            epsilon_max=config.controller.epsilon_max,
            lambda_explore=config.controller.lambda_explore,
            theta_idk=config.controller.theta_idk,
        )
        self.controller.seed(seed)

        # Episode history
        self.history = []

    def _get_state(self) -> int:
        """Get the current state index from the environment."""
        pos = tuple(self.env.unwrapped.agent_pos)
        return self.mapper.get_index(pos)

    def train_episode(self, env_seed=None) -> dict:
        """Run one episode. Returns metrics dict.

        Args:
            env_seed: Seed for env.reset(). Required for MiniGrid envs
                where wall positions change per seed (e.g. FourRooms).

        Metrics:
            episode_reward: Total reward for the episode.
            steps: Number of steps taken.
            unique_states: Number of unique states visited.
            mean_confidence: Mean C(s) over visited states.
            mean_uncertainty: Mean U(s) over visited states.
            idk_count: Number of steps where idk_flag was True.
            change_detected: Whether change was detected this episode.
            reached_goal: Whether the agent reached the goal.
        """
        obs, info = self.env.reset(seed=env_seed)
        s = self._get_state()

        episode_reward = 0.0
        steps = 0
        visited_states = set()
        confidences = []
        uncertainties = []
        idk_count = 0

        done = False
        while not done:
            visited_states.add(s)

            # Select action (pass agent direction for greedy V_explore)
            agent_dir = self.env.unwrapped.agent_dir
            action, confidence, idk_flag = self.controller.select_action(
                s, MOVEMENT_ACTIONS, agent_dir=agent_dir
            )
            confidences.append(confidence)
            uncertainties.append(self.meta_sr.uncertainty(s))
            if idk_flag:
                idk_count += 1

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Get next state
            s_next = self._get_state()

            # Update SR (returns delta_M for meta-SR)
            delta_M = self.sr.update(s, s_next, reward)

            # Update meta-SR
            self.meta_sr.observe(s, delta_M)

            episode_reward += reward
            steps += 1
            s = s_next

        metrics = {
            "episode_reward": episode_reward,
            "steps": steps,
            "unique_states": len(visited_states),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
            "idk_count": idk_count,
            "change_detected": self.meta_sr.detect_change(),
            "reached_goal": bool(episode_reward > 0),
        }
        self.history.append(metrics)
        return metrics

    def train(self, n_episodes: int, log_every: int = 50, progress: bool = True,
              env_seed=None):
        """Full training loop.

        Args:
            n_episodes: Number of episodes to train.
            log_every: Print summary every N episodes.
            progress: Show tqdm progress bar.
            env_seed: Seed for env.reset() each episode. Required for
                MiniGrid envs where wall positions change per seed.
        """
        iterator = trange(n_episodes, desc="Training") if progress else range(n_episodes)

        for ep in iterator:
            metrics = self.train_episode(env_seed=env_seed)

            if progress and ep % log_every == 0:
                iterator.set_postfix(
                    reward=f"{metrics['episode_reward']:.2f}",
                    conf=f"{metrics['mean_confidence']:.3f}",
                    unc=f"{metrics['mean_uncertainty']:.3f}",
                    states=metrics["unique_states"],
                )

    def get_uncertainty_map(self) -> np.ndarray:
        """Return current U(s) for all states."""
        return self.meta_sr.all_uncertainties()

    def get_confidence_map(self) -> np.ndarray:
        """Return current C(s) for all states."""
        return self.meta_sr.all_confidences()

    def get_value_map(self) -> np.ndarray:
        """Return current V(s) for all states."""
        return self.sr.all_values()
