from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
from threading import Thread
import time

import numpy as np
from isaacsim.core.api import World
from tasks.pick_place_task import PickPlace
from controllers.pick_place import PickPlaceController
from cube_observer import CubeObserver#, cube_observation

my_world = World(stage_units_in_meters=1.0)
target_position = np.array([-0.3, -0.3, 0])
target_position[2] = 0.04 / 2.0
pp_task = PickPlace(name="fr3_pick_place", target_position=target_position)
my_world.add_task(pp_task)
my_world.reset()
my_fr3 = my_world.scene.get_object("fr3_robot")

pp_controller = PickPlaceController(name="controller", robot_articulation=my_fr3, gripper=my_fr3.gripper)
task_params = my_world.get_task("fr3_pick_place").get_params()
articulation_controller = my_fr3.get_articulation_controller()

i = 0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            pp_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()

        actions = pp_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0, 0]),
        )
        cube_observation = CubeObserver(observations=observations, task_params=task_params)
        print(observations)
        print(f"Phase: {pp_controller.get_current_event()}")
        if (cube_observation == True) & (pp_controller.get_current_event() > 4):
            actions = pp_controller.grip(action="open")
            pp_controller.change_event(0)
        if pp_controller.is_done():
            print("done")
        articulation_controller.apply_action(actions)

simulation_app.close()
