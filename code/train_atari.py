import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

import matplotlib.pyplot as plt

if __name__ == "__main__":

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
    env = gym.wrappers.Monitor(env, "recordings")

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

    # determine the number of steps over which we decay epsilon
    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])

    # store the rewards and losses for plotting
    episode_rewards = [0.0]
    episode_loss = []

    state = env.reset()

    average_loss = 0
    num_actions = 1
    for t in range(hyper_params["num-steps"]):
        # linearly interpolate the value of epsilon for the first `eps_timesteps` steps
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )

        #  select random action if sample is less equal than eps_threshold, else the agent acts greedily
        sample = random.random()
        if sample <= eps_threshold:
            action = env.action_space.sample()
        else:
            action = agent.act(np.array(state))

        # take step in env
        next_state, reward, done, _ = env.step(action)

        # add state, action, reward, next_state, float(done) to replay buffer - cast done to float
        agent.memory.add(state, action, reward, next_state, float(done))

        # update the state for the next iteration
        state = next_state

        # add reward to episode_reward
        episode_rewards[-1] += reward

        # if the episode has terminated, reset the environment
        if done:
            state = env.reset()
            episode_rewards.append(0.0)
            episode_loss.append(average_loss / num_actions)
            average_loss = 0
            num_actions = 1

        # governs how often we update the weights of our network (e.g. via backprop)
        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            average_loss += agent.optimise_td_loss()
            num_actions += 1

        # governs how often we update the target network to the current network
        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        # print a progress update at the end of episode
        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_100ep_loss = round(np.mean(episode_loss[-101:-1]), 8)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("mean 100 episode loss: {}".format(mean_100ep_loss))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")

    # plot the rewards
    plt.plot(episode_rewards)
    # set the plot labels
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    # set the plot title
    plt.title("Episode Reward")
    # show the plot
    plt.savefig("episode-reward.png", format="png")
    plt.clf()

    # plot the loss
    plt.plot(episode_loss)
    # set the plot labels
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    # set the plot title
    plt.title("Episode Loss")
    # show the plot
    plt.savefig("episode-loss.png", format="png")

    # save the final model weights
    agent.save_model("./model.pt")
