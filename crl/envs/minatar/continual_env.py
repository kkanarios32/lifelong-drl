# Adapted from https://github.com/qlan3/gym-games
import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    pass
except:
    import logging

    logging.warning("Cannot import seaborn.Will not be able to train from pixel observations.")

from envs.minatar.environment import Environment


class BaseEnv(gym.Env):
    metadata = {"render.modes": ["human", "array"]}

    def __init__(
        self,
        game,
        display_time=50,
        use_minimal_action_set=True,
        use_minimal_observation=False,
        sticky_action_prob=0.1,
        seed=None,
        **kwargs,
    ):
        self.game_name = game
        self.sticky_action_prob = sticky_action_prob
        self.display_time = display_time
        self.use_minimal_observation = use_minimal_observation

        self.game_kwargs = kwargs
        self.seed(seed)

        if use_minimal_action_set:
            self.action_set = self.game.minimal_action_set()
        else:
            self.action_set = list(range(self.game.num_actions()))

        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(0.0, 1.0, shape=self.game.state_shape(), dtype=bool)

    def step(self, action):
        action = self.action_set[action]
        reward, done = self.game.act(action)
        return self.game.state(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.game = Environment(
            env_name=self.game_name,
            sticky_action_prob=self.sticky_action_prob,
            use_minimal_observation=self.use_minimal_observation,
            **self.game_kwargs,
        )
        self.game.reset()
        return self.game.state(), {}

    def seed(self, seed=None):
        self.game = Environment(
            env_name=self.game_name,
            sticky_action_prob=self.sticky_action_prob,
            use_minimal_observation=self.use_minimal_observation,
            **self.game_kwargs,
        )
        return seed

    def render(self, mode="human"):
        if mode == "array":
            return self.game.state()
        elif mode == "human":
            self.game.display_state(self.display_time)

    def close(self):
        if self.game.visualized:
            self.game.close_display()
        return 0


def CL_envs_cycle(idx):
    idx = idx % 2
    all_envs = ["breakout", "space_invaders", "freeway"]
    sample_env = all_envs[idx]
    return BaseEnv(
        sample_env,
    )


def CL_envs_func(env_name):
    if env_name == "all":
        all_envs = ["breakout", "space_invaders", "freeway"]
        sample_env = np.random.choice(all_envs)
        return BaseEnv(
            sample_env,
        )
    else:
        raise NotImplementedError
