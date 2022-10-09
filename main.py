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
from minihack import RewardManager
import wandb

# 1. Start a new run
wandb.init(project="dqn")
# wandb.init(project="dqn", mode="disabled")

hyper_params = {
    "seed": 42,  # which seed to use
    "env": "MiniHack-ExploreMaze-Easy-v0",  # name of the game
    "replay-buffer-size": int(1e6),  # replay buffer size
    "learning-rate": 1e-4,  # learning rate for Adam optimizer
    "discount-factor": 0.99,  # discount factor
    "num-steps": int(1e6),  # total number of steps to run the environment for
    "num-episodes":
    100000,  # total number of episodes to run the environment for
    "batch-size": 256,  # number of transitions to optimize at the same time
    "learning-starts": 10000,  # number of steps before learning starts
    "learning-freq": 1,  # number of iterations between every optimization step
    "use-double-dqn": True,  # use double deep Q-learning
    "target-update-freq":
    500,  # number of iterations between every target network update
    "eps-start": 1.0,  # e-greedy start threshold
    "eps-end": 0.1,  # e-greedy end threshold
    "eps-fraction": 0.1,  # fraction of num-steps over which to decay epsilon
    "print-freq": 100,
    "save-freq": 250,
}

wandb.config = hyper_params

MOVE_ACTIONS = tuple(nethack.CompassDirection)  # h,j,k,l,y,u,b,n
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
               penalty_time=-0.005,
               penalty_step=-0.1)

env._max_episode_steps = 30

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
)

# actions[50] == PRAY
state = env.reset()  # each reset generates a new environment instance

t = 0
num_episodes = 0
average_loss = 0
num_actions = 1
episode_rewards = [0.0]
episode_loss = []

#check if there is a saved model in ./models. If there is, load it and set the t variable to the number of steps it has trained for and the num_episodes to the number of episodes it has trained for
try:
    # get the name of the latest model
    model_name = natsorted(os.listdir('./models'))[-1]
    # load the model

    checkpoint = agent.load_model('./models/' + model_name)
    agent.policy_network.load_state_dict(checkpoint['model_state_dict'])
    agent.update_target_network()
    agent.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    num_episodes = checkpoint['episode']
    t = checkpoint['t']

    agent.memory._storage = checkpoint['storage']
    agent.memory._next_idx = checkpoint['next_idx']

    print(f'Loaded model from step {t} and episode {num_episodes}')
except:
    print('No saved model to load. Starting new training run.')
    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params['use-double-dqn'],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        gamma=hyper_params['discount-factor'],
    )
    t = 0
    num_episodes = 0
    pass

eps_timesteps = hyper_params['eps-fraction'] * float(hyper_params['num-steps'])


# 3. Log gradients and model parameters
wandb.watch(agent.policy_network)

while t < hyper_params['num-steps'] and num_episodes < hyper_params[
        'num-episodes']:

    fraction = min(1.0, float(t) / eps_timesteps)
    eps_threshold = hyper_params['eps-start'] + fraction * (
        hyper_params['eps-end'] - hyper_params['eps-start'])

    #  select random action if sample is less equal than eps_threshold, else the agent acts greedily
    sample = random.random()
    if sample <= eps_threshold:
        action = env.action_space.sample()
    else:
        action = agent.act(np.array(state).reshape(4, 9,9))

    next_state, reward, done, info = env.step(action)

    # add state, action, reward, next_state, float(done) to replay buffer - cast done to float
    agent.memory.add(state, action, reward, next_state, float(done))

    # update the state for the next iteration
    state = next_state

    # add reward to episode_reward
    episode_rewards[-1] += reward

    if done:
        state = env.reset()
        # 4. Log metrics to visualize performance
        wandb.log({"loss": average_loss/num_actions, "reward": episode_rewards[-1], "epsilson": eps_threshold})

        episode_rewards.append(0.0)
        episode_loss.append(average_loss / num_actions)
        average_loss = 0
        num_actions = 1
        num_episodes += 1


    # governs how often we update the weights of our network (e.g. via backprop)
    if (t > hyper_params['learning-starts']
            and t % hyper_params['learning-freq'] == 0):
        average_loss += agent.optimise_td_loss()
        num_actions += 1

    # governs how often we update the target network to the current network
    if (t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0):
        agent.update_target_network()


    # print a progress update at the end of episode
    if (done and hyper_params["print-freq"] is not None
            and num_episodes % hyper_params["print-freq"] == 0):
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 4)
        mean_100ep_loss = round(np.mean(episode_loss[-101:-1]), 8)
        print("********************************************************")
        print("steps: {}".format(t))
        print("episodes: {}".format(num_episodes))
        print("mean 100 episode reward: {}".format(mean_100ep_reward))
        print("mean 100 episode loss: {}".format(mean_100ep_loss))
        print("% time spent exploring: {}".format(int(100 * eps_threshold)))
        print("********************************************************")

    if (done and hyper_params["save-freq"] is not None
            and num_episodes % hyper_params["save-freq"] == 0):
        agent.save_model("./models/model_{}.pth".format(num_episodes),
                         num_episodes, t)

    t += 1