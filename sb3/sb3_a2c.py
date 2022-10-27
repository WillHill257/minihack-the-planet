import gym
import minihack
import time

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C, DQN
from sb3_contrib import RecurrentPPO
import math

import torch
from torch import nn

import os


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":

    env_id = 'CartPole-v1'
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # wrap env with a VecMonitor
    env = VecMonitor(env)

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        ent_coef=0.,
        verbose=1,
    )

    # model = DQN(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     learning_rate=4e-3,
    #     batch_size=128,
    #     buffer_size=10000,
    #     learning_starts=1000,
    #     gamma=0.98,
    #     target_update_interval=600,
    #     train_freq=16,
    #     gradient_steps=8,
    #     exploration_fraction=0.2,
    #     exploration_final_eps=0.07,
    #     policy_kwargs=dict(net_arch=[256, 256]))

    # model = RecurrentPPO(
    #     "MlpLstmPolicy",
    #     env,
    #     n_steps= 32,
    #     batch_size= 256,
    #     gae_lambda= 0.8,
    #     gamma= 0.98,
    #     n_epochs= 20,
    #     ent_coef= 0.0,
    #     policy_kwargs= dict(
    #                         ortho_init=False,
    #                         activation_fn=nn.ReLU,
    #                         lstm_hidden_size=64,
    #                         enable_critic_lstm=True,
    #                         net_arch=[dict(pi=[64], vf=[64])]
    #                     ),
    #     verbose=1,
    # )
    try:
        model.learn(total_timesteps=int(1.2e5), progress_bar=True)
        model.save("a2c")
    except KeyboardInterrupt:
        pass

    env = DummyVecEnv([make_env(env_id, 0)])
    obs = env.reset()

    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs, ), dtype=bool)
    for _ in range(2000):
        # action, lstm_states = model.predict(obs,
        #                                     state=lstm_states,
        #                                     episode_start=episode_starts,
        #                                     deterministic=True)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        env.render()

        episode_starts = dones

        if dones:
            obs = env.reset()