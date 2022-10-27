from pydoc import cli
import gym
import minihack
import time

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO

import torch
from torch import nn


def evaluate_lstm(model, env_id):
    num_envs = 1
    env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=DummyVecEnv)
    obs = env.reset()

    # cell and hidden state of the LSTM
    lstm_states = None
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs, ), dtype=bool)
    for _ in range(2000):
        action, lstm_states = model.predict(obs,
                                            state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=True)

        obs, rewards, dones, info = env.step(action)

        episode_starts = dones
        env.render()

        if dones:
            time.sleep(1)
            obs = env.reset()


class MiniHackExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(MiniHackExtractor, self).__init__(observation_space,
                                                features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            n_input_channels = subspace.shape[0]

            if len(subspace.shape) < 4:
                # This is a 2D image, so expand the dimensions
                n_input_channels = 1

            if key == 'chars':
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                total_concat_size += 16 * 21 * 79
            elif key == 'chars_crop':
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                total_concat_size += 16 * 9 * 9

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # Add a channel to the 2d observation
            if len(observations[key].shape) < 4:
                observations[key] = observations[key].unsqueeze(1)
            encoded_tensor_list.append(extractor(observations[key] / 128.0))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


def make_env(env_id, rank, seed=0, args={}):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id, **args)
        env._max_episode_steps = 30
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env_id = "MiniHack-River-v0"
    args = {
        'observation_keys': ['chars', 'chars_crop'],
        'penalty_time': -0.005,
        'penalty_step': -0.1
    }

    num_cpu = 12  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i, args=args) for i in range(num_cpu)])

    # wrap env with a VecMonitor
    env = VecMonitor(env)

    model = RecurrentPPO("MultiInputLstmPolicy",
                         env,
                         learning_rate=0.0013,
                         verbose=1,
                         gamma=0.9698,
                         ent_coef=0.000178,
                         max_grad_norm=1.8557,
                         n_epochs=1,
                         batch_size=32,
                         clip_range=0.1,
                         n_steps=32,
                         gae_lambda=0.992,
                         policy_kwargs=dict(
                             features_extractor_class=MiniHackExtractor,
                             ortho_init=True,
                             lstm_hidden_size=64,
                             net_arch=[{
                                 "pi": [64, 64],
                                 "vf": [64, 64]
                             }]))

    try:
        model.learn(total_timesteps=int(2e5))
    except KeyboardInterrupt:
        pass

    evaluate_lstm(model, env_id)