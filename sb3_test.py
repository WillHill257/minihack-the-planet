import gym
import minihack
import time

import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
from torch import nn


class CustomCombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space,
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
                    nn.Conv2d(16, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                total_concat_size += 8 * 21 * 79
            elif key == 'chars_crop':
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                total_concat_size += 8 * 9 * 9

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
            encoded_tensor_list.append(extractor(observations[key]/128.0))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)

env = gym.make("MiniHack-River-v0",
               observation_keys=['chars', 'chars_crop'],
               penalty_time=-0.005,
               penalty_step=-0.1)
env._max_episode_steps = 30

model = A2C("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=50000, progress_bar=True)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

    time.sleep(0.1)

    if dones:
        obs = env.reset()