#! /usr/bin/env python3

# Copyright (C) 2022 Statistical Machine Learning and Motor Control Group (SLMC)
# Authors: Joao Moura (maintainer)
# email: joao.moura@ed.ac.uk

# This file is part of iiwa_optas package.

# iiwa_optas is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# iiwa_optas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
import numpy as np
import math
import os
import time

os.system('bash -c "source /opt/ros/noetic/setup.bash"')
os.system('bash -c "source /home/iiwa-kuka/software/andreas_ws/devel/setup.bash"')

import sys

sys.path.append("/home/iiwa-kuka/software/andreas_miiniconda/ot_policy")
# ros
import rospkg
import rospy
import tf2_ros
import tf2_geometry_msgs  # needed just because yes
import actionlib
from cob_srvs.srv import SetString, SetStringRequest
import collections


# messages
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, Image

# policy related
import torch
import dill
import hydra
from cot_policy.real_world.multistep_obs import MultiStepObsWrapper
from diffusion_policy.common.pytorch_util import dict_apply


def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


class PolicyNode(object):
    """docstring for PolicyActionServer."""

    def __init__(self, name):
        # initialization message
        self.name = name
        rospy.loginfo(f"{self.name}: Initializing class")
        #################################################################################
        ## get parameters:
        #################################################################################
        self.cmd_twist_action_server_name = rospy.get_param(
            "~cmd_twist_action_server_name", "demo_collection"
        )
        self.link_ee = rospy.get_param("~link_ee", "tool_link")
        self.link_ref = rospy.get_param("~link_ref", "ws_ref")
        self._freq = rospy.get_param(
            "~freq", 10
        )  # should be the same as the control frequency
        self.dt = 1.0 / self._freq
        self.num_inference_steps = rospy.get_param("~num_inference_steps", 3) + 1
        self.checkpoint_path = rospy.get_param(
            "~checkpoint_path",
<<<<<<< HEAD
            "/home/iiwa-kuka/software/andreas_miiniconda/exps/models/pybullet_pushing/latest.ckpt",
=======
            "/home/iiwa-kuka/software/andreas_miiniconda/exps/models/real_pushing_cot_policy/latest.ckpt",
>>>>>>> origin/main
        )
        #################################################################################
        # tf2 stuff
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        #################################################################################
        # initialize variables
        self.twist_output = Twist()
        self.output_dir = rospkg.RosPack().get_path("iiwa_optas") + "/output"
        self.image_H = 240
        self.image_W = 240
        self.img_dict = {
            "camera_0": np.zeros((3, self.image_H, self.image_W)),
            "camera_1": np.zeros((3, self.image_H, self.image_W)),
        }
        self.ee_pos = np.zeros(2)
        self.n_obs_steps = 2
        self.action_horizon = 8
        self.current_action_step = 0
<<<<<<< HEAD
        # self.action = np.ones((self.action_horizon, 2)) * self.ee_pos #self.read_ee_from_tf()
        self.action = np.zeros((self.action_horizon, 2))
=======
        self.action = np.ones((self.action_horizon, 2)) * self.read_ee_from_tf()
>>>>>>> origin/main
        self.device = torch.device("cuda:0")
        #################################################################################
        # declare subscribers
        # # RealSense camera topic
        # self._rs_sub = rospy.Subscriber(
        #     "/camera/color/image_raw", Image, self.read_img_callback
        # )
        # ROS-Pybullet camera topic
<<<<<<< HEAD
        # self._rs_sub_camera_0 = rospy.Subscriber(
        #     "/camera1/color/image_raw",
        #     Image,
        #     self.read_img_callback_camera_0,
        # )
        # self._rs_sub_camera_1 = rospy.Subscriber(
        #     "/camera2/color/image_raw",
        #     Image,
        #     self.read_img_callback_camera_1,
        # )
        self._rs_sub_camera_0 = rospy.Subscriber(
            "/rpbi/camera_1/colour/image",
=======
        self._rs_sub_camera_0 = rospy.Subscriber(
            "/camera1/color/image_raw",
>>>>>>> origin/main
            Image,
            self.read_img_callback_camera_0,
        )
        self._rs_sub_camera_1 = rospy.Subscriber(
<<<<<<< HEAD
            "/rpbi/camera_2/colour/image",
=======
            "/camera2/color/image_raw",
>>>>>>> origin/main
            Image,
            self.read_img_callback_camera_1,
        )
        #################################################################################
        # declare publishers
        self.twist_pub = rospy.Publisher(
            name="/policy_twist", data_class=Twist, queue_size=10
        )
        #################################################################################
        # load policy and obs wrapper
        # time.sleep(5)
        self.obs_wrapper = MultiStepObsWrapper(n_obs_steps=self.n_obs_steps)
        self.obs_wrapper.reset(self.get_obs_dict())
        self.policy = self.load_policy()
        # assert self.action_horizon <= self.policy.n_action_steps
        #################################################################################
        # timer creation for policy execution
        dur = rospy.Duration(1.0 / self._freq)
        self.timer = rospy.Timer(dur, self.timer_callback)
        self.t0 = rospy.Time.now().to_sec()

    def timer_callback(self, event):
        """Publish the policy output"""
        # register new observation
        self.obs_wrapper.add_obs(self.get_obs_dict())
        # check if we are within an action sequence
        if self.current_action_step >= (self.action_horizon - 1):
            # get stacked observation
            obs = self.obs_wrapper.get_obs(self.n_obs_steps)
            # get action
            self.get_action(obs)
            # reset action step
            self.current_action_step = 0
        else:
            # increment action step
            self.current_action_step += 1

        # publish action
<<<<<<< HEAD
        action = self.action[self.current_action_step]
        output_twist = self.get_twist_from_action(action)
        self.twist_pub.publish(output_twist)
        print(f"Action: {action}")
=======

        next_ee_pos = self.action[self.current_action_step]
        current_ee_pos = self.read_ee_from_tf()

        velocity_command = (next_ee_pos - current_ee_pos) / self.dt

        output_twist = self.get_twist_from_action(velocity_command)
        self.twist_pub.publish(output_twist)
        print(f"Action: {velocity_command}")
>>>>>>> origin/main

    def get_twist_from_action(self, action):
        twist_output = Twist()
        twist_output.linear.x = clip(action[0], -0.5, 0.5)
        twist_output.linear.y = clip(action[1], -0.5, 0.5)
        # twist_output.linear.z = clip(action[2], -0.5, 0.5)
        # twist_output.angular.x = clip(action[3], -0.5, 0.5)
        # twist_output.angular.y = clip(action[4], -0.5, 0.5)
        # twist_output.angular.z = clip(action[5], -0.5, 0.5)
        return twist_output

    def get_action(self, obs):
        # publish zero action to make the robot wait until new action is calculated
        zero_action = np.zeros(2)
        self.twist_pub.publish(self.get_twist_from_action(zero_action))

        # get action from policy
        np_obs_dict = dict(obs)

        # device transfer
        obs_dict = dict_apply(
            np_obs_dict,
            lambda x: torch.from_numpy(x).to(device=self.device).unsqueeze(0),
        )
        # run policy
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())

        self.action = np_action_dict["action"][0]
        if not np.all(np.isfinite(self.action)):
            print(self.action)
            raise RuntimeError("Nan or Inf action")

    def get_obs_dict(self):
        self.ee_pos = self.read_ee_from_tf()
        obs_dict = {
            "camera_0": self.img_dict["camera_0"],
            "camera_1": self.img_dict["camera_1"],
            "ee_pos": self.ee_pos,
        }
        return obs_dict

    def read_img_callback_camera_0(self, msg):
        # Get image dimensions and encoding from the message
        width = msg.width
        height = msg.height
        num_channels = 3

        # Convert the image data to a NumPy array
        # msg.data is a flat byte array, so we need to reshape it
        image_data = np.frombuffer(msg.data, dtype=np.uint8)
        image_data = image_data.reshape((height, width, num_channels))

        # Reshape to training shape (120 x 160)
        from PIL import Image

        img = Image.fromarray(image_data)  # Convert to PIL Image
        img_resized = img.resize((self.image_W, self.image_H))  # Resize
<<<<<<< HEAD
        image = np.array(img_resized, dtype=np.float32) / 255
        image = np.moveaxis(image, -1, 0)


        # image = np.moveaxis(image_data, -1, 0)

        self.img_dict["camera_0"] = image
        # self.ee_pos = self.read_ee_from_tf()
=======
        image = np.array(img_resized, dtype=np.uint8)
        image = np.moveaxis(image, -1, 0)
        # image = np.moveaxis(image_data, -1, 0)

        self.img_dict["camera_0"] = image
        self.ee_pos = self.read_ee_from_tf()
>>>>>>> origin/main

    def read_img_callback_camera_1(self, msg):
        # Get image dimensions and encoding from the message
        width = msg.width
        height = msg.height
        num_channels = 3

        # Convert the image data to a NumPy array
        # msg.data is a flat byte array, so we need to reshape it
        image_data = np.frombuffer(msg.data, dtype=np.uint8)
        image_data = image_data.reshape((height, width, num_channels))
        # Reshape to training shape (120 x 160)
        from PIL import Image

        img = Image.fromarray(image_data)  # Convert to PIL Image
        img_resized = img.resize((self.image_W, self.image_H))  # Resize
<<<<<<< HEAD
        image = np.array(img_resized, dtype=np.float32) / 255
=======
        image = np.array(img_resized, dtype=np.uint8)
>>>>>>> origin/main
        image = np.moveaxis(image, -1, 0)

        self.img_dict["camera_1"] = image

<<<<<<< HEAD
        # self.ee_pos = self.read_ee_from_tf()
=======
        self.ee_pos = self.read_ee_from_tf()
>>>>>>> origin/main

    def read_ee_from_tf(self, source_frame="iiwa_link_ee", parent_frame="world"):
        try:
            trans = self.tf_buffer.lookup_transform(
                "world", "iiwa_link_ee", rospy.Time()
            )
            self.pos_x = trans.transform.translation.x
            self.pos_y = trans.transform.translation.y
            self.pos_z = trans.transform.translation.z
            self.rot_x = trans.transform.rotation.x
            self.rot_y = trans.transform.rotation.y
            self.rot_z = trans.transform.rotation.z
            self.rot_w = trans.transform.rotation.w
            self.ee_read = np.array(
                [
                    self.pos_x,
                    self.pos_y,
                    self.pos_z,
                    self.rot_x,
                    self.rot_y,
                    self.rot_z,
                    self.rot_w,
                ],
                dtype=np.float32,
            )
            self.ee_2d_pos = np.array([self.pos_x, self.pos_y], dtype=np.float32)
            # rospy.logwarn("end-effector pos x: %s", self.ee_read)
            return self.ee_2d_pos  # Return only 2D position
        except ():
            rospy.logfatal(
                f"{self.name}: Unable to find the pose from {source_frame} to {parent_frame}!"
            )

    def load_policy(self):
        # load workspace
        payload = torch.load(open(self.checkpoint_path, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=self.output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        policy.n_action_steps = self.action_horizon
        policy.num_inference_steps = self.num_inference_steps
        print(f"Policy loaded with {policy.num_inference_steps} inference steps")

        # set device
        policy.to(self.device)
        policy.eval()

        return policy


if __name__ == "__main__":
    # Initialize node
    rospy.init_node("policy", anonymous=True)
    # Initialize node class
    PolicyNode(rospy.get_name())
    # executing node
    rospy.spin()
