# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils import distance_metrics
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import *
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper


class FrankaRmpFlowExampleScript:
    def __init__(self):
        self._robot_num = 3

        self._rmpflow = None
        self._articulation_rmpflow = []

        self._articulation = []
        self._target = None

        self._script_generator = None

    def load_example_assets(self, num):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """
        self._robot_num = num

        robot_prim_path = "/FR3"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/FR3/fr3.usd"
        for i in range(self._robot_num):
            add_reference_to_stage(path_to_robot_usd, robot_prim_path+f"_{i}")
            gripper = ParallelGripper(
                end_effector_prim_path=f"/FR3_{i}/fr3_leftfinger",
                joint_prim_names=["fr3_finger_joint1", "fr3_finger_joint2"],
                joint_opened_positions=np.array([0.04, 0.04]),
                joint_closed_positions=np.array([0, 0]),
                action_deltas=np.array([0.04, 0.04]),
            )
            self._articulation.append(SingleManipulator(
                prim_path=f"/FR3_{i}",
                name=f"fr3_robot_{i}",
                position=np.array([0, 0.8*i, 0]),
                end_effector_prim_name="fr3_leftfinger",
                gripper=gripper,
            ))

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = SingleXFormPrim(
            "/World/target",
            scale=[0.04, 0.04, 0.04],
            position=np.array([0.4, 0, 0.25]),
            orientation=euler_angles_to_quats([0, np.pi, 0]),
        )

        self._goal_block = DynamicCuboid(
            name="Cube",
            position=np.array([0.4, 0, 0.025]),
            prim_path="/World/pick_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
        )
        self._ground_plane = GroundPlane("/World/Ground")

        # Return assets that were added to the stage so that they can be registered with the core.World
        return *self._articulation, self._target, self._goal_block, self._ground_plane

    def setup(self, num):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        self._robot_num = num
        # Set a camera view that looks good
        set_camera_view(eye=[2, 0.8, 1], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        self._rmpflow = lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(os.path.dirname(__file__), "./rmpflow/fr3_robot_description.yaml"),
            rmpflow_config_path=os.path.join(os.path.dirname(__file__), "./rmpflow/fr3_rmpflow_config.yaml"),
            urdf_path=os.path.join(os.path.dirname(__file__), "./rmpflow/fr3.urdf"),
            end_effector_frame_name="gripper_center",
            maximum_substep_size=0.00334,
        )
        for i in range(self._robot_num):
            self._articulation_rmpflow.append(ArticulationMotionPolicy(self._articulation[i], self._rmpflow))
        # Create a script generator to execute my_script().
        self._script_generator = self.my_script()

    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        # Start the script over by recreating the generator.
        self._script_generator = self.my_script()

    """
    The following two functions demonstrate the mechanics of running code in a script-like way
    from a UI-based extension.  This takes advantage of Python's yield/generator framework.  

    The update() function is tied to a physics subscription, which means that it will be called
    one time on every physics step (usually 60 frames per second).  Each time it is called, it
    queries the script generator using next().  This makes the script generator execute until it hits
    a yield().  In this case, no value need be yielded.  This behavior can be nested into subroutines
    using the "yield from" keywords.
    """

    def update(self, step: float):
        try:
            result = next(self._script_generator)
        except StopIteration:
            return True

    def my_script(self):
        translation_target, orientation_target = self._target.get_world_pose()

        for i in range(self._robot_num):
            yield from self.close_gripper_franka(self._articulation[i])
            lower_translation_target, _ = self._goal_block.get_world_pose()
            lower_translation_target[1] -= 0.8*i
            translation_target[0] = lower_translation_target[0]
            translation_target[1] = lower_translation_target[1]
            # Notice that subroutines can still use return statements to exit.  goto_position() returns a boolean to indicate success.
            success = yield from self.goto_position(
                translation_target, orientation_target, self._articulation[i], self._rmpflow, timeout=200
            )

            '''if not success:
                print("Could not reach target position")
                print(translation_target)
                return'''

            yield from self.open_gripper_franka(self._articulation[i])
            
            self._target.set_world_pose(lower_translation_target, orientation_target)

            success = yield from self.goto_position(
                lower_translation_target, orientation_target, self._articulation[i], self._rmpflow, timeout=250
            )

            yield from self.close_gripper_franka(self._articulation[i], close_position=np.array([0.02, 0.02]), atol=0.006)

            high_translation_target = np.array([0.4, 0, 0.4])
            self._target.set_world_pose(high_translation_target, orientation_target)

            success = yield from self.goto_position(
                high_translation_target, orientation_target, self._articulation[i], self._rmpflow, timeout=200
            )

            next_translation_target = np.array([0.4, 0.6, 0.4])
            self._target.set_world_pose(next_translation_target, orientation_target)

            success = yield from self.goto_position(
                next_translation_target, orientation_target, self._articulation[i], self._rmpflow, timeout=200
            )

            next_translation_target = np.array([0.4, 0.6, 0.04])
            self._target.set_world_pose(next_translation_target, orientation_target)

            success = yield from self.goto_position(
                next_translation_target, orientation_target, self._articulation[i], self._rmpflow, timeout=200
            )

            yield from self.open_gripper_franka(self._articulation[i])

            next_translation_target[2] = 0.4
            self._target.set_world_pose(next_translation_target, orientation_target)

            success = yield from self.goto_position(
                next_translation_target, orientation_target, self._articulation[i], self._rmpflow, timeout=200
            )

            next_translation_target = np.array([0.4, 0, 0.4])
            self._target.set_world_pose(next_translation_target, orientation_target)

            success = yield from self.goto_position(
                next_translation_target, orientation_target, self._articulation[i], self._rmpflow, timeout=200
            )

    ################################### Functions

    def goto_position(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=0.1,
        timeout=500,
    ):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """

        articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 60)
        rmpflow.set_end_effector_target(translation_target, orientation_target)

        for i in range(timeout):
            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, translation_target)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)

            done = trans_dist < translation_thresh and rot_dist < orientation_thresh

            if done:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 60)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            yield ()

        return False

    def open_gripper_franka(self, articulation):
        open_gripper_action = ArticulationAction(np.array([0.04, 0.04]), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully opened.
        while not np.allclose(articulation.get_joint_positions()[7:], np.array([0.04, 0.04]), atol=0.001):
            yield ()

        return True

    def close_gripper_franka(self, articulation, close_position=np.array([0, 0]), atol=0.001):
        # To close around the cube, different values are passed in for close_position and atol
        open_gripper_action = ArticulationAction(np.array(close_position), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully closed.
        while not np.allclose(articulation.get_joint_positions()[7:], np.array(close_position), atol=atol):
            yield ()

        return True
