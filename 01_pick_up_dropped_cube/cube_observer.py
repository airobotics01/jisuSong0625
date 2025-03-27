last_zpos = 0
def is_cube_fallen_odd(current_pos, target_pos, threshold=0.05) -> bool:
    global last_zpos
    if (current_pos[2] < 0.025) & (current_pos[2] == last_zpos):
        dist = sum((current_pos[i] - target_pos[i]) ** 2 for i in range(2)) ** 0.5
        return dist > threshold
    else:
        last_zpos = current_pos[2]
        return False

def CubeObserver(observations, task_params, threshold=0.05) -> bool:
    current_position = observations[task_params["cube_name"]["value"]]["position"]
    target_position=observations[task_params["cube_name"]["value"]]["target_position"]
    if (is_cube_fallen_odd(current_position, target_position, threshold)):
        return True
    else:
        return False