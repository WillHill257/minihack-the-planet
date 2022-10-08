from gym import spaces
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        # 3 conv layers, and 2 fully-connected layers, as specified in the original paper (in Nature)
        self.conv1 = nn.Conv2d(observation_space.shape[0], 8, 3)
        self.linear1 = nn.Linear(8 * self.conv1.shape[1] * self.conv1.shape[2], 64)
        self.linear2 = nn.Linear(64, action_space.n)

    def forward(self, x):
        # forward pass, using relu activations

        # 3 conv layers
        x = F.relu(self.conv1(x))

        # flatten from the multidimensional feature map from the conv layers, to a single dimension for the fully-connected layers
        x = x.view(-1, 8 * self.conv1.shape[1] * self.conv1.shape[2])

        # 2 fully-connected layers
        x = F.relu(self.linear1(x))
        x = self.linear2(x)  # no activation on the output layer
        return x
