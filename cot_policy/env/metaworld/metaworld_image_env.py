from gym import spaces, Wrapper
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np
import mujoco


class MetaworldImageEnv(Wrapper):
    def __init__(
        self,
        env_name: str,
        shape_meta,
        img_height: int = 128,
        img_width: int = 128,
        cameras=("corner2",),
        camera_angle="high",
        env_kwargs=None,
    ):
        if env_kwargs is None:
            env_kwargs = {}
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env_name}-goal-observable"](
            **env_kwargs
        )
        env._freeze_rand_vec = False
        self.env = env

        if camera_angle == "high":
            self.env.model.cam_pos[2] = [0.75, 0.2, 0.65]
            self.env.model.cam_fovy[2] = 35
        elif camera_angle == "low":
            self.env.model.cam_pos[2] = [0.75, 0.1, 0.4]
            self.env.model.cam_fovy[2] = 45
        else:
            print("Wrong camera angle. Using default high angle.")
            self.env.model.cam_pos[2] = [0.75, 0.2, 0.65]
            self.env.model.cam_fovy[2] = 35

        self.img_width = img_width
        self.img_height = img_height
        obs_meta = shape_meta["obs"]
        # self.rgb_outputs = list(obs_meta["corner_rgb"]["shape"])
        # self.lowdim_outputs = list(obs_meta["robot_states"]["shape"])

        self.rgb_outputs = []
        self.lowdim_outputs = []
        obs_key_shapes = dict()
        for key, attr in obs_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            typee = attr.get("type", "low_dim")
            if typee == "rgb":
                self.rgb_outputs.append(key)
            elif typee == "low_dim":
                self.lowdim_outputs.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {typee}")

        self.cameras = cameras
        self.render_mode = "rgb_array"

        super().__init__(env)
        obs_space_dict = {}
        for key in self.rgb_outputs:
            obs_space_dict[key] = spaces.Box(
                low=0, high=1, shape=obs_key_shapes[key], dtype=np.float32
            )
        for key in self.lowdim_outputs:
            obs_space_dict[key] = spaces.Box(
                low=self.env.observation_space.low[[0, 1, 2, 3]],
                high=self.env.observation_space.high[[0, 1, 2, 3]],
                shape=obs_key_shapes[key],
                dtype=np.float32,
            )
        # obs_space_dict["obs_gt"] = env.observation_space
        self.env.observation_space = spaces.Dict(obs_space_dict)
        self.env.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            dtype=self.env.action_space.dtype,
            shape=self.env.action_space.shape,
        )

        self.viewer = None

    def step(self, action):
        obs_gt, reward, terminated, truncated, info = self.env.step(action)
        obs_gt = obs_gt.astype(np.float32)
        info["obs_gt"] = obs_gt
        self.obs_gt = obs_gt

        next_obs = self._get_obs()

        terminated = info["success"] == 1
        reward = float(
            terminated
        )  # We don't care about the reward, only if the task is completed succesfully
        done = terminated or truncated

        return next_obs, reward, done, info

    def reset(self, seed=None, options=None):
        if self.viewer is None:
            self.viewer = OffScreenViewer(
                self.env.model,
                self.env.data,
                self.img_width,
                self.img_height,
                self.env.mujoco_renderer.max_geom,
                self.env.mujoco_renderer._vopt,
            )
        if seed is not None:
            self._seed = seed
        self.seed(self._seed)
        obs_gt, info = super().reset(seed=seed)
        # self.seed(self._seed)

        obs_gt = obs_gt.astype(np.float32)
        info["obs_gt"] = obs_gt
        self.obs_gt = obs_gt

        obs = self._get_obs()

        return obs

    def _get_obs(self):
        obs = {}
        # Rename 'robot_states' to 'agent_pos'
        # obs["robot_states"] = np.concatenate((self.obs_gt[:4], self.obs_gt[18:22]))
        # obs["obs_gt"] = obs_gt
        obs["robot_states"] = self.obs_gt[:4]

        image_dict = {}
        for camera_name in self.cameras:
            image_obs = self.render(camera_name=camera_name, mode="all")
            image_dict[camera_name] = image_obs

        # Replace 'corner_rgb' key with 'image' for all RGB outputs
        for key in self.rgb_outputs:
            # Use 'image' as the key instead of the formatted camera name key
            obs[key] = (
                image_dict[f"{key[:-4]}2"].transpose(2, 0, 1).astype(np.float32) / 255.0
            )  # [
            # ::-1
            # ]  #   # Assuming 'corner' -> 'corner2' flip

        return obs

    def render(self, camera_name=None, mode="all"):
        if camera_name is None:
            camera_name = self.cameras[0]
        cam_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        return self.viewer.render(render_mode=mode, camera_id=cam_id)[::-1]

    def set_task(self, task):
        self.env.set_task(task)
        self.env._partially_observable = False

    def seed(self, seed):
        self._seed = seed
        self.env.seed(seed)
        self.env.np_random = np.random.RandomState(seed)
        np.random.seed(seed)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    shape_meta = {
        "action": {"shape": [4]},
        "obs": {
            "robot_states": {"shape": [4], "type": "low_dim"},
            "corner_rgb": {"shape": [3, 224, 224], "type": "rgb"},
        },
    }

    wrapper = MetaworldImageEnv(
        "soccer-v2", shape_meta=shape_meta, img_height=224, img_width=224
    )
    seed = 1000
    wrapper.seed(seed)
    wrapper.env.model.cam_pos[2] = [0.75, 0.2, 0.7]
    wrapper.env.model.cam_fovy[2] = 45
    obs = wrapper.reset()
    obs, _, _, _ = wrapper.step(wrapper.action_space.sample())
    # Example (3, 224, 224) numpy array

    # Step 1: Ensure correct shape (224, 224, 3)
    image_array = np.transpose(obs["corner_rgb"], (1, 2, 0))

    # Step 2: Convert to uint8 if in range [0, 1]
    if image_array.dtype == np.float32 and image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)

    # Step 3: Save as image
    image = Image.fromarray(image_array)
    image.save(f"output_image_{seed}.png")
