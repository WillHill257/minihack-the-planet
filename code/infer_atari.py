import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # these are just copied from the training script -> most are meaningless here, just included for completeness sake
    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps over which to decay epsilon
        "print-freq": 1,
    }

    # set the random seeds
    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    # use the environment wrappers
    # when reset the environment, take a random number of no-op actions before training further (so we don't always start in the same state)
    env = NoopResetEnv(env, noop_max=30)
    # only return every 4th frame, with accumulated rewards. perform the same action for the intermediate frames
    env = MaxAndSkipEnv(env, skip=4)
    # if we run out of lives, treat as the end of an episode
    env = EpisodicLifeEnv(env)
    # if the environment is fixed until a "fire" action is taken, take this action now -> i.e. click the "start" button for the game
    env = FireResetEnv(env)
    # resize the input frame to 84x84, as expected by the DQN
    env = WarpFrame(env)
    # change the image shape to the format pytorch expects
    env = PyTorchFrame(env)
    # set the reward to -1 for negative scores, and 1 for positive scores
    # done so that consistent hyper-params can be used to train across multiple games where the scoring magnitudes (therefore, rewards) differ
    env = ClipRewardEnv(env)
    # the number of consecutive frames from the environment to stack to use as input to the DQN
    env = FrameStack(env, 4)
    # just a wrapper to periodically record the training episode for visual purposes
    env = gym.wrappers.Monitor(env, "inference_recordings")

    # create the replay buffer of a certain length
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        hyper_params["use-double-dqn"],
        hyper_params["learning-rate"],
        hyper_params["batch-size"],
        hyper_params["discount-factor"],
    )
    agent.load_model("./model.pt")

    # run inference
    state = env.reset()

    # set the network up for evaluations
    agent.policy_network.eval()
    done = False
    while not done:
        # always sample an action from the agent
        action = agent.act(np.array(state))

        # take step in env
        next_state, reward, done, _ = env.step(action)

        # update the state
        state = next_state
