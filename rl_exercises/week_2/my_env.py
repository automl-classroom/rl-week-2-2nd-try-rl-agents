from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.state = 0  # always start in state 0

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state."""
        self.state = 0
        return int(self.state), {}

    def step(self, action):
        """Perform one step in the environment."""
        if not self.action_space.contains(action):
            raise RuntimeError(f"Invalid action: {action}")

        self.state = action  # action is either 0 or 1-> go to that state
        reward = float(action)  # reward is equal to the action
        terminated = False
        truncated = False
        info = {}

        return int(self.state), reward, terminated, truncated, info

    def get_reward_per_action(self):
        """Returns the reward for each (state, action) pair as a matrix."""
        R = np.zeros((2, 2), dtype=float)
        for s in range(2):
            for a in range(2):
                R[s, a] = float(a)  # reward is equal to te action
        return R

    def get_transition_matrix(self):
        """Returns the transition probability matrix T[s, a, s']"""
        T = np.zeros((2, 2, 2), dtype=float)
        for s in range(2):
            T[s, 0, 0] = 1.0  # action 0 -> always go to state 0
            T[s, 1, 1] = 1.0  # action 1 -> always go to state 1
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        pass
