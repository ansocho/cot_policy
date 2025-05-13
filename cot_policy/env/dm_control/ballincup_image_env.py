from gym import spaces, Wrapper
import numpy as np
import cv2
from shimmy.registration import DM_CONTROL_SUITE_ENVS
import gymnasium as gym
from collections import OrderedDict


class BallInCupImageEnv(Wrapper):
    def __init__(self, render_size=96):

        env_name = "ball_in_cup-catch"
        env = gym.make(
            "dm_control/" + env_name + "-v0",
            render_mode="rgb_array",
        )

        dm_env = env.env.env._env

        self.render_cache = None
        self.render_size = render_size
        self.env = env
        self.dm_env = dm_env
        self.env.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.env.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32
                ),
                "agent_pos": spaces.Box(
                    low=0, high=np.inf, shape=(4,), dtype=np.float32
                ),
            }
        )
        self._seed = None
        self.seed()

        super().__init__(env)

    def _get_obs(self):
        img = self.dm_env.physics.render(
            camera_id=0, width=self.render_size, height=self.render_size
        )

        agent_pos = np.concatenate(
            [self.last_obs["position"][0:2], self.last_obs["velocity"][0:2]], axis=-1
        )
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {"image": img_obs, "agent_pos": agent_pos}

        self.render_cache = img

        return obs

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            self._get_obs()

        return self.render_cache

    def reset(self):

        time_step = self.dm_env.reset()
        self.last_obs = time_step.observation  # Store the initial observation

        return self._get_obs()

    def step(self, action):
        time_step = self.dm_env.step(action)
        self.last_obs = time_step.observation  # Update with the latest observation

        obs = self._get_obs()  # Get the updated observation
        # Extract reward, done, and info
        reward = time_step.reward
        done = time_step.last()
        info = {}
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        task = self.dm_env.task
        task._random = np.random.RandomState(seed)


# a = BallInCupImageEnv()
# a.seed(100)
# time_step = a.reset()
# for i in range(100):
#     obs, reward, done, info = a.step(np.random.randn(2))
#     a.render()
#     img = a.render_cache
#     image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(f"rendered_image_{i}.png", image_bgr)
