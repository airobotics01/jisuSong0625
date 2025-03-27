import os
from typing import Optional

import isaacsim.core.api.tasks as tasks
import numpy as np
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper


class PickPlace(tasks.PickPlace):
    def __init__(
        self,
        name: str = "fr3_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=np.array([0.04, 0.04, 0.04]),
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")
        asset_path = assets_root_path + "/Isaac/Robots/Franka/FR3/fr3.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/FR3")
        gripper = ParallelGripper(
            end_effector_prim_path="/World/FR3/fr3_leftfinger",
            joint_prim_names=["fr3_finger_joint1", "fr3_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0, 0]),
            action_deltas=np.array([0.04, 0.04]),
        )
        manipulator = SingleManipulator(
            prim_path="/World/FR3",
            name="fr3_robot",
            end_effector_prim_name="fr3_leftfinger",
            gripper=gripper,
        )
        joints_default_positions = np.zeros(9)
        joints_default_positions[7], joints_default_positions[8] = 0.04, 0.04
        manipulator.set_joints_default_state(positions=joints_default_positions)
        return manipulator
