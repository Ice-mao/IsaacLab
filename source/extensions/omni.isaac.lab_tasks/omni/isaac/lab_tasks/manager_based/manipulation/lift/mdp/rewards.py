# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer, Camera
from omni.isaac.lab.utils.math import combine_frame_transforms


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    reward = (1 - torch.tanh(object_ee_distance / std)) * (robot.period < 2) + 1.0 * (robot.period == 2)
    return reward


def object_is_dropped(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    #robot.period >= 2
    reward = (robot.period == 2) * torch.sum(env.action_manager.action[:,5] > 0.5, dim=-1) / env.num_envs
    # reward = (robot.period >= 2) * torch.tanh(object_ee_distance / std)
    return reward

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    threshold:float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    if command_name=='object_pose_stay':
        command = env.command_manager.get_command('object_pose')
    else:
        command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    if command_name=='object_pose_middle':
        robot.period[torch.logical_and(distance < threshold, robot.period == 0)] = 1
        reward = ((object.data.root_pos_w[:, 2] > minimal_height) *
                  (1 - torch.tanh(distance / std)) * (robot.period == 0) +
                   (1 - torch.tanh(torch.ones_like(distance)*(threshold/std))) * (robot.period >= 1))
    if command_name=="object_pose":
        robot.period[torch.logical_and(distance < threshold, robot.period == 1)] = 2
        reward = ((object.data.root_pos_w[:, 2] > minimal_height) *
                  (1.0 - torch.tanh(distance / std)) * (robot.period == 1) +
                  (object.data.root_pos_w[:, 2] > minimal_height) *
                  (1.0 - torch.tanh(distance / std)) * (robot.period == 2)) # 稳定最终状态
                #   + (1 - torch.tanh(torch.ones_like(distance)*(threshold/std))) * (robot.period >= 2))
    if command_name=='object_pose_stay':
        robot.period[torch.logical_and(distance < threshold, robot.period == 2)] = 3
        reward = ((object.data.root_pos_w[:, 2] > minimal_height) *
                  (1.0 - torch.tanh(distance / std)) * (robot.period == 2)
                  + (1 - torch.tanh(torch.ones_like(distance)*(threshold/std))) * (robot.period == 3))

    return reward

def debug_pp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
)-> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    camera: Camera = env.scene[camera_cfg.name]
    # torch.save(camera.data.output["rgb"], 'images_tensor_rgb.pt')
    # torch.save(camera.data.output["distance_to_image_plane"], 'images_tensor_depth.pt')
    reward = 1 - torch.tanh(torch.ones(32))
    return reward.to('cuda')
