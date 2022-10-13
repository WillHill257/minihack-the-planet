import gym
import time

from model import ActorCritic
from multiprocessing_env import SubprocVecEnv, VecPyTorch, VecPyTorchFrameStack
from wrappers import *
import torch

import minihack
from nle import nethack
import argparse
from natsort import natsorted
import os


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("A2C experiments for Atari games")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Environment
    parser.add_argument("--env", type=str, default="MiniHack-ExploreMaze-Easy-v0", help="name of the game")
    # Core A2C parameters
    parser.add_argument("--actor-loss-coefficient", type=float, default=1.0, help="actor loss coefficient")
    parser.add_argument("--critic-loss-coefficient", type=float, default=0.5, help="critic loss coefficient")
    parser.add_argument("--entropy-loss-coefficient", type=float, default=0.01, help="entropy loss coefficient")
    parser.add_argument("--lr", type=float, default=7e-4, help="learning rate for the RMSprop optimizer")
    parser.add_argument("--alpha", type=float, default=0.99, help="alpha term the RMSprop optimizer")
    parser.add_argument("--eps", type=float, default=1e-5, help="eps term for the RMSprop optimizer")  # instead of 1e-3 due to different RMSprop implementation in PyTorch than Tensorflow
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm of gradients")
    parser.add_argument("--num_steps", type=int, default=5, help="number of forward steps")
    parser.add_argument("--num-envs", type=int, default=1, help="number of processes for environments")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--num-frames", type=int, default=int(10e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--log-dir", type=str, default="logs", help="where to save log files")
    parser.add_argument("--save-freq", type=int, default=10000, help="updates between saving models (default 0 => no save)")
    # Reporting
    parser.add_argument("--print-freq", type=int, default=1000, help="evaluation frequency.")
    return parser.parse_args()

def make_env(seed):
    env = gym.make(args.env,
            observation_keys=["glyphs_crop"],
            actions=ALL_ACTIONS,
            penalty_time=-0.005,
            penalty_step=-0.1)
    env._max_episode_steps = 30
    # env.seed(seed)
    env = StateSpaceFrame(env)
    env = ChannelWrapper(env)
    # env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    return env



if __name__ == '__main__':
    args = parse_args()

    ALL_ACTIONS = tuple(nethack.CompassDirection)

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    env = make_env(args.seed)

    actor_critic = ActorCritic(env.observation_space, env.action_space).to(device)

    try:
        # get the name of the latest model
        model_name = natsorted(os.listdir('A2C/models'))[-1]
        # load the model
        print("Loading model...")
        checkpoint = torch.load('A2C/models/' + model_name)
        actor_critic.load_state_dict(checkpoint)
        actor_critic.eval()
    except:
        print('No saved model to load')
        exit()



    observation = env.reset()

    while True:
        # check if observation has a shape attribute
        if hasattr(observation, 'shape'):
            observation /= 5991.0
        else:
            observation = torch.FloatTensor(np.array(observation) / 5991.0).unsqueeze(0).to(device)
        actor, value = actor_critic(observation)

        action = np.argmax(actor.probs.detach().cpu())
        next_observation, reward, done, infos = env.step(action.unsqueeze(0))

        if done:
            break

        env.render()
        time.sleep(0.2)