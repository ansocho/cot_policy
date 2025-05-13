import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def stack_last_n_obs(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepObsWrapper:
    def __init__(
        self,
        n_obs_steps,
    ):
        self.n_obs_steps = n_obs_steps
        self.obs = deque(maxlen=n_obs_steps + 1)

    def reset(self, obs):

        self.obs_dict = obs
        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        obs = self.get_obs(self.n_obs_steps)
        return obs

    def add_obs(self, obs):
        """
        Adds an observation to the queue. To be used at each timestep of the action horizon.
        Call get_obs() to get the stacked observations and calculate the next action sequence
        """
        self.obs.append(obs)

    def get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert len(self.obs) > 0
        result = dict()
        for key in self.obs_dict.keys():
            result[key] = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
        return result
