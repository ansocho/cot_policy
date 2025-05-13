from gym import spaces, Wrapper
import numpy as np
import cv2
import gym
from collections import OrderedDict
import d4rl


class Maze2dEnv(Wrapper):
    def __init__(self, env_name, target=None):

        self.env_name = env_name
        env = gym.make(
            env_name,
        )
        if target is None:
            self.target = env.get_target()
        else:
            self.target = target
        self.env = env
        self.env.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
            }
        )
        self._seed = None
        self.seed()

        super().__init__(env)

    def _get_obs(self):

        obs = {"agent_pos": np.concatenate([self.state, self.target], axis=-1)}

        return obs

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)

    def reset(self):
        self.state = self.env.reset()
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.state, reward, done, _ = self.env.step(
            action
        )  # Update with the latest observation

        obs = self._get_obs()  # Get the updated observation
        # Extract reward, done, and info
        info = {}
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.env.seed(seed)


# a = BallInCupImageEnv()
# a.seed(100)
# time_step = a.reset()
# for i in range(100):
#     obs, reward, done, info = a.step(np.random.randn(2))
#     a.render()
#     img = a.render_cache
#     image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(f"rendered_image_{i}.png", image_bgr)
