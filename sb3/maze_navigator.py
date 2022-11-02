import time
import numpy as np

# direction agent is facing
# to look right: (direc + 1) % 4
# to look left:  (direc - 1) % 4
N = 0 + 4
E = 1 + 4
S = 2 + 4
W = 3 + 4

directions = [
    np.array([-1, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([0, -1]),
]  # up, right, down, left (N, E, S, W)


def glyph_pos(glyphs, glyph):
    glyph_positions = np.where(np.asarray(glyphs) == glyph)
    assert len(glyph_positions) == 2
    if glyph_positions[0].shape[0] == 0:
        return None
    # (row, col)
    return np.array([glyph_positions[0][0], glyph_positions[1][0]], dtype=np.uint8)


def is_valid_pos(pos, dim_of_observations):
    return (
        pos[0] >= 0
        and pos[1] >= 0
        and pos[0] < dim_of_observations[0]
        and pos[1] < dim_of_observations[1]
    )


def policy_update_direction_of_travel(chars_crop, current_direction):
    """
    our policy:
    - if there is an open space to the right, turn right (+ orientation)
    - if there is not an open space to the right
        - if there is an open space in front, move there
        - else, turn orientation left

    returns: new_direction_facing
    """

    valid_spaces = [ord("#"), ord("+"), ord("<")]

    agent_position = glyph_pos(chars_crop, ord("@"))

    # is there an open space to our right
    pos = tuple(agent_position + directions[(current_direction + 1) % 4])
    if is_valid_pos(pos, chars_crop.shape) and chars_crop[pos] in valid_spaces:
        # move here
        return (current_direction + 1) % 4 + 4

    # can't move right
    # is forward open
    pos = tuple(agent_position + directions[current_direction % 4])
    if is_valid_pos(pos, chars_crop.shape) and chars_crop[pos] in valid_spaces:
        # move here
        return (current_direction) % 4 + 4

    # can't move forward, check left
    pos = tuple(agent_position + directions[(current_direction - 1) % 4])
    if is_valid_pos(pos, chars_crop.shape) and chars_crop[pos] in valid_spaces:
        # move here
        return (current_direction - 1) % 4 + 4

    # go back the way we came
    return (current_direction + 2) % 4 + 4


def navigate_maze(env, observations, visualise=False):
    """
    take in the environment in which to act, and then follow a turn-right policy until get to the end of the maze
    we will do this with the following ideas:
        - only look at the cropped portion (only need to make local decisions)
        - the following chars are of interest: | (vertical wall), - (horizontal wall), # (open corridor)
    our policy:
        - if there is an open space to the right, turn right (+ orientation)
        - if there is not an open space to the right
            - if there is an open space in front, move there
            - else, turn orientation left
    """

    # keep track of the direction we're facing - always start facing up
    direction_facing = N

    # get the observations
    observations = observations["chars_crop"][0]

    # visualise
    if visualise:
        env.render()

    # loop until get to the end of the maze
    end_of_maze = False
    while not end_of_maze:
        # use the current state to inform actions
        direction_facing = policy_update_direction_of_travel(
            observations, direction_facing
        )

        # check if this direction is the exit
        agent_position = glyph_pos(observations, ord("@"))
        end_of_maze = observations[
            tuple(agent_position + directions[direction_facing % 4])
        ] == ord("+")

        # convert the direction want to travel in to an actual action
        action = [direction_facing % 4]  # env.MOVE_ACTIONS[direction_facing % 4]

        # perform the action
        observations, rewards, dones, infos = env.step(action)
        observations = observations["chars_crop"][0]

        # visualise
        if visualise:
            env.render()
            time.sleep(1 / 24)

    # we are now at the end of the maze, repeat the last action we took (which was to walk forwards into the doorway)
    observations, rewards, dones, infos = env.step(action)
