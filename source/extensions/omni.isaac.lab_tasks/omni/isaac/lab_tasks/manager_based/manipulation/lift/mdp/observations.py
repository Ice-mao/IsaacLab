# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    # only return the position of frame object w.r.t. frame robot
    return object_pos_b

def target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    # only return the position of frame object w.r.t. frame robot
    return des_pos_b

def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)
def image_flatten(
    env: ManagerBasedRLEnv,
    data_type: str = "rgb",
    normalize: bool = True,
) -> torch.Tensor:
    """
    image_flatten() inherits from the image()

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor = env.scene.sensors["camera"]

    # obtain the input image
    if data_type == "rgbd":
        images_rgb = sensor.data.output["rgb"]
        images_depth = sensor.data.output["distance_to_image_plane"]
        if normalize:
            # RGB:image的数据围绕0分布进行
            images_rgb = images_rgb.float() / 255.0
            mean_tensor = torch.mean(images_rgb, dim=(1, 2), keepdim=True)
            images_rgb -= mean_tensor
            # Depth:
            images_depth[images_depth == float("inf")] = 0
        images = torch.cat((images_rgb, images_depth), dim=3)
    else:
        images = sensor.data.output[data_type]
    images = images.permute(0, 3, 1, 2)  # 转换为 (N, C, H, W)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            # image的数据围绕0分布进行
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone().reshape(images.size(0), -1)
