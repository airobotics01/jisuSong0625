from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from isaacsim.core.api.objects import DynamicCylinder
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name

class ObjectPickPlace(ABC, BaseTask):
    """[summary]

    Args:
        name (str): [description]
        object_initial_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        object_initial_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        object_size (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        object_type: str,
        object_initial_position: Optional[np.ndarray] = None,
        object_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        object_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._target_object = None
        self._object = None
        self._object_type = None
        self._object_initial_position = object_initial_position
        self._object_initial_orientation = object_initial_orientation
        self._target_position = target_position
        self._object_size = object_size
        if self._object_size is None:
            self._object_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if self._object_initial_position is None:
            self._object_initial_position = np.array([0.3, 0.3, 0.3]) / get_stage_units()
        if self._object_initial_orientation is None:
            self._object_initial_orientation = np.array([1, 0, 0, 0])
        if self._target_position is None:
            self._target_position = np.array([-0.3, -0.3, 0]) / get_stage_units()
            self._target_position[2] = self._object_size[2] / 2.0
        self._target_position = self._target_position + self._offset
        return

    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        #name = f"/World/{object_name}"
        object_prim_path = find_unique_string_name(
            initial_name="/World/Cylinder", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        object_name = find_unique_string_name(initial_name="object_name", is_unique_fn=lambda x: not self.scene.object_exists(x))
        self._object = scene.add(
            DynamicCylinder(
                name=object_name,
                position=self._object_initial_position,
                orientation=self._object_initial_orientation,
                prim_path=object_prim_path,
                scale=self._object_size,
                radius=0.4,
                height=0.8,
                color=np.array([0, 0, 1]),
            )
        )
        self._task_objects[self._object.name] = self._object
        self._robot = self.set_robot()
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return

    @abstractmethod
    def set_robot(self) -> None:
        raise NotImplementedError

    def set_params(
        self,
        object_position: Optional[np.ndarray] = None,
        object_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
    ) -> None:
        if target_position is not None:
            self._target_position = target_position
        if object_position is not None or object_orientation is not None:
            self._object.set_local_pose(translation=object_position, orientation=object_orientation)
        return

    def get_params(self) -> dict:
        params_representation = dict()
        position, orientation = self._object.get_local_pose()
        params_representation["object_position"] = {"value": position, "modifiable": True}
        params_representation["object_orientation"] = {"value": orientation, "modifiable": True}
        params_representation["target_position"] = {"value": self._target_position, "modifiable": True}
        params_representation["object_name"] = {"value": self._object.name, "modifiable": False}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        object_position, object_orientation = self._object.get_local_pose()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        return {
            self._object.name: {
                "position": object_position,
                "orientation": object_orientation,
                "target_position": self._target_position,
            },
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
            },
        }

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        return

    def post_reset(self) -> None:
        from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        return

    def calculate_metrics(self) -> dict:
        """[summary]"""
        raise NotImplementedError

    def is_done(self) -> bool:
        """[summary]"""
        raise NotImplementedError