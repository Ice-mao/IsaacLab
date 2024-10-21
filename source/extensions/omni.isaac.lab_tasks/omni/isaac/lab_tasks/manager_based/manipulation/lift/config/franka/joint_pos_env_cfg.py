# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
import omni.isaac.lab.sim as sim_utils

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/home/sangfor/Documents/IsaacLab/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/Collected_stardust_pro/stardust_pro.usd",
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
                    "joint2": 0.6,
                    "joint3": 0.3,
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
                ),
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", ],
                    effort_limit=12.0,
                    velocity_limit=2.61,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "finger": ImplicitActuatorCfg(
                    joint_names_expr=["tool_leftfingerjoint", "tool_rightfingerjoint"],
                    effort_limit=100.0,
                    velocity_limit=0.2,
                    stiffness=2e3,
                    damping=1e2,
                ),

            },
        )

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["tool_leftfingerjoint", "tool_rightfingerjoint"],
            open_command_expr={"tool_leftfingerjoint": 0.0, "tool_rightfingerjoint": -0.0},
            close_command_expr={"tool_leftfingerjoint": 0.05, "tool_rightfingerjoint": -0.05},
        )

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 0.03], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/stardust_pro/base_footprint",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/stardust_pro/tool_leftfinger_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, -0.05, -0.0], #TODO: center from tool_link, initial: 0.1034
                    ),
                ),
            ],
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
