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

hyper_params = {
    "seed": 42,  # which seed to use
    "env": "MiniHack-ExploreMaze-Easy-v0",  # name of the game
    "replay-buffer-size": int(5e3),  # replay buffer size
    "learning-rate": 1e-4,  # learning rate for Adam optimizer
    "discount-factor": 0.99,  # discount factor
    "num-steps": int(1e6),  # total number of steps to run the environment for
    "num-episodes": 1000,  # total number of episodes to run the environment for
    "batch-size": 256,  # number of transitions to optimize at the same time
    "learning-starts": 10000,  # number of steps before learning starts
    "learning-freq": 1,  # number of iterations between every optimization step
    "use-double-dqn": True,  # use double deep Q-learning
    "target-update-freq": 100,  # number of iterations between every target network update
    "eps-start": 1.0,  # e-greedy start threshold
    "eps-end": 0.01,  # e-greedy end threshold
    "eps-fraction": 0.1,  # fraction of num-steps over which to decay epsilon
    "print-freq": 20,
    "save-freq": 10,
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

env = gym.make(hyper_params["env"],
               observation_keys=["glyphs_crop"],
               actions=ALL_ACTIONS,
               penalty_time=-0.005)

env._max_episode_steps = 50

env = StateSpaceFrame(env)

env = FrameStack(env, 4)

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
mean_reward = 0.0

while True:
    action = agent.act(np.array(state).reshape(4, 9,9))

    next_state, reward, done, info = env.step(action)

    # update the state for the next iteration
    state = next_state

    steps += 1
    mean_reward += (reward - mean_reward) / steps

    if done:
        time_delta = timeit.default_timer() - start_time
        print("Final reward:", reward)
        print("End status:", info["end_status"].name)
        print("Mean reward:", mean_reward)

        sps = steps / time_delta
        print(f"Steps: {steps}. SPS: {sps}")
        break

    env.render()
    time.sleep(0.2)