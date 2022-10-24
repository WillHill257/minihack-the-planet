import gym
import minihack
import time

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C

import torch
from torch import nn


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
    env_id = "LunarLander-v2"
    num_cpu = 10  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # wrap env with a VecMonitor
    env = VecMonitor(env)

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    linear_schedule = lambda progress_remaining: progress_remaining * 0.00083
    model = A2C(
        "MlpPolicy",
        env,
        n_steps=5,
        gamma=0.995,
        learning_rate=linear_schedule,
        ent_coef=0.00001,
        verbose=1,
    )
    try:
        model.learn(total_timesteps=int(5e5))
    except KeyboardInterrupt:
        pass

    env = DummyVecEnv([make_env(env_id, 0)])
    obs = env.reset()
    for _ in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

        if dones:
            obs = env.reset()