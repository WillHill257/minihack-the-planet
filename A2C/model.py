import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        observation_shape = observation_space.shape
        n_actions = action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )

        # self.gru = nn.GRUCell(32 * 7 * 7, 256)
        self.linear = nn.Linear(8 * 9 * 9, 256)
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.critic.bias, 0.0)

        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.constant_(self.actor.bias, 0.0)

    def forward(self, x):
        # Expected 4-dimensional input for 4-dimensional weight [32, 36, 8, 8], but got 3-dimensional input of size [1, 36, 9] instead
        x = self.features(x)
        x = x.view(-1, 8 * 9 * 9)
        x = self.linear(x)
        x = F.relu(x)
        # hx = self.gru(x, hx)
        return Categorical(logits=self.actor(x)), self.critic(x)