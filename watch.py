import gym
import minihack
from nle import nethack
from DQN.agent import DQNAgent
from DQN.replay_buffer import ReplayBuffer
from DQN.wrappers import StateSpaceFrame, FrameStack
import random
import numpy as np
import os
from natsort import natsorted
import time
import timeit
import hashlib

hyper_params = {
    "seed": 42,  # which seed to use
    "env": "MiniHack-ExploreMaze-Easy-Mapped-v0",  # name of the game
    "replay-buffer-size": int(1e6),  # replay buffer size
    "learning-rate": 1e-4,  # learning rate for Adam optimizer
    "discount-factor": 0.99,  # discount factor
    "num-steps": int(1e6),  # total number of steps to run the environment for
    "num-episodes":
    100000,  # total number of episodes to run the environment for
    "batch-size": 32,  # number of transitions to optimize at the same time
    "learning-starts": 10000,  # number of steps before learning starts
    "learning-freq": 4,  # number of iterations between every optimization step
    "use-double-dqn": True,  # use double deep Q-learning
    "target-update-freq":
    10000,  # number of iterations between every target network update
    "eps-start": 1.0,  # e-greedy start threshold
    "eps-end": 0.02,  # e-greedy end threshold
    "eps-fraction": 0.1,  # fraction of num-steps over which to decay epsilon
    "print-freq": 100,
    "save-freq": 1000,
}


MOVE_ACTIONS = tuple(nethack.CompassDirection) # h,j,k,l,y,u,b,n
# ALL_ACTIONS = MOVE_ACTIONS + (
#     nethack.Command.ZAP,  # z
#     nethack.Command.PRAY, # M-p
#     nethack.Command.APPLY, # a
#     nethack.Command.PICKUP, # ,

#     nethack.Command.WEAR, # W

#     nethack.Command.FIRE, # f
#     nethack.Command.RUSH, # g

# )

ALL_ACTIONS = MOVE_ACTIONS

replay_buffer = ReplayBuffer(hyper_params['replay-buffer-size'])

explored = {}

def bad_actions(env, prev, act, curr):
    if act > 7:
        return -1
    return 0


def hash_pair(point):
    # Cantors pairing function
    x, y = point
    return (x**2 + 3 * x + 2 * x * y + y + y**2) / 2


def move_right(env, prev, act, curr):
    player_loc = np.squeeze(np.where(curr[1] == 64))
    if player_loc.size != 2:
        return 0
    return 0.001 * player_loc[1]


def count_based(env, prev, act, curr):
    # b = curr[0].view(np.uint8)
    # hash = hashlib.sha1(b).hexdigest()
    player_loc = np.squeeze(np.where(curr[1] == 64))
    if player_loc.size != 2:
        return 0

    hash = hash_pair(tuple(player_loc))
    if hash not in explored:
        explored[hash] = 0
    explored[hash] += 1
    return 0.01 / explored[hash]


def new_states(env, prev, act, curr):

    if (prev[1] == 32).sum() > (curr[1] == 32).sum():
        return 0.1

    return 0

env = gym.make(
    hyper_params["env"],
    observation_keys=["glyphs",'glyphs_crop'],
    #    actions=ALL_ACTIONS,
    penalty_time=-0.001,
    penalty_step=-0.1)

env.reward_manager.add_custom_reward_fn(bad_actions)
env.reward_manager.add_custom_reward_fn(count_based)
env.reward_manager.add_custom_reward_fn(new_states)
env.reward_manager.add_custom_reward_fn(move_right)

env._max_episode_steps = 50


agent = DQNAgent(
    env.observation_space,
    env.action_space,
    replay_buffer,
    use_double_dqn=hyper_params['use-double-dqn'],
    lr=hyper_params['learning-rate'],
    batch_size=hyper_params['batch-size'],
    gamma=hyper_params['discount-factor'],
    # device='cpu',
)


# actions[50] == PRAY
state = env.reset() # each reset generates a new environment instance

#check if there is a saved model in ./models. If there is, load it and set the t variable to the number of steps it has trained for and the num_episodes to the number of episodes it has trained for
try:
    # get the name of the latest model
    model_name = natsorted(os.listdir('./models'))[-1]
    # load the model
    print("Loading model...")
    checkpoint = agent.load_model('./models/' + model_name)
    agent.policy_network.load_state_dict(checkpoint['model_state_dict'])
    agent.update_target_network()
    agent.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

    agent.memory._storage = checkpoint['storage']
    agent.memory._next_idx = checkpoint['next_idx']
except:
    print('No saved model to load')
    exit()

start_time = timeit.default_timer()
steps = 0
total_reward = 0


while True:
    action = agent.act(state)

    next_state, reward, done, info = env.step(action)

    # update the state for the next iteration
    state = next_state

    steps += 1
    total_reward += reward

    print(f'Action: {action}, Reward: {reward:.5f}')
    # convert torch tensor q_values to normal python list and put it on the cpu
    q_values = np.squeeze(agent.q_values.detach().cpu().numpy()).tolist()
    print(np.round(q_values, 5))

    if done:
        time_delta = timeit.default_timer() - start_time
        print("Final reward:", reward)
        print("End status:", info["end_status"].name)
        print("Total reward:", total_reward)
        print("Mean reward:", total_reward/steps)

        sps = steps / time_delta
        print(f"Steps: {steps}. SPS: {sps}")
        break

    env.render()
    time.sleep(0.2)