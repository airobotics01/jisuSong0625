from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from tasks.pick_place_task import PickPlace
from controllers.pick_place import PickPlaceController

_tasks = []
_num_of_tasks = 3
_controllers = []
_robots = []
_cube_names = []

my_world = World(stage_units_in_meters=1.0)
target_position = np.array([-0.3, -0.3, 0])
target_position[2] = 0.04 / 2.0
rotate = np.sin(np.radians(90)/2)
#pp_task = PickPlace(name="fr3_pick_place", object_initial_orientation=np.array([0, -rotate, 0, -rotate]), target_position=target_position)
#my_world.add_task(pp_task)
for j in range(_num_of_tasks):
    my_world.add_task(PickPlace(name="my_multiple_robots_task_"+str(j), offset=np.array([0, (j*2)-_num_of_tasks, 0])))

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
            picking_position=observations[task_params["object_name"]["value"]]["position"],
            placing_position=observations[task_params["object_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0, 0]),
        )
        if pp_controller.is_done():
            print("done")
        articulation_controller.apply_action(actions)

simulation_app.close()
