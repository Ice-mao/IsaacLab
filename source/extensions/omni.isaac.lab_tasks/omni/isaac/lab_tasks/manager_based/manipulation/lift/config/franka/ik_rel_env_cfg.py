# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
import omni.isaac.lab.sim as sim_utils

@configclass
class FrankaCubeLiftEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/sangfor/Documents/IsaacLab/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/Collected_stardust_pro",
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "left_wheel_joint": 0.0,
                    "right_wheel_joint": 0.0,
                    "joint1": 0.0,
                    "joint2": 0.0,
                    "joint3": 0.0,
                    "joint4": 0.0,
                    "joint5": 0.0,
                    "tool_leftfingerjoint": 0.0,
                    "tool_rightfingerjoint": -0.0,
                },
                pos=(0.0, 0.0, 0.07),
                rot=(1, 0, 0, 0.0),
            ),
            actuators={
                "wheel": ImplicitActuatorCfg(
                    joint_names_expr=["left_wheel_joint", "right_wheel_joint", ],
                    effort_limit=87.0,
                    velocity_limit=2.175,
                    stiffness=80.0,
                    damping=4.0,
                    # friction=0.0001,
                ),
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", ],
                    effort_limit=87.0,
                    velocity_limit=2.0944,
                    stiffness=80.0,
                    damping=4.0,
                    # friction=0.0001,
                ),
                "finger": ImplicitActuatorCfg(
                    joint_names_expr=["tool_leftfingerjoint", "tool_rightfingerjoint"],
                    effort_limit=20.0,
                    velocity_limit=0.1,
                    stiffness=2e3,
                    damping=1e2,
                ),

            },
        )

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            body_name="tool_leftfinger_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            # On many robots, end-effector frames are fictitious frames that do not have a corresponding rigid body.
            # In such cases, it is easier to define this transform w.r.t. their parent rigid body.
            # For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the “panda_hand” frame.
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
