from gym import spaces
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class DQN(nn.Module):

    def __init__(self, observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        super().__init__()
        # 3 conv layers, and 2 fully-connected layers, as specified in the original paper (in Nature)

        # 4 x 21 x 79
        self.conv_glyph1 = nn.Conv2d(1, 16, 3, padding=1)
        # 16 x 21 x 79
        self.conv_glyph2 = nn.Conv2d(16, 16, 3, padding=1)
        # 16 x 21 x 79
        self.conv_glyph3 = nn.Conv2d(16, 16, 3, padding=1)
        # 16 x 21 x 79
        self.conv_glyph4 = nn.Conv2d(16, 16, 3, padding=1)
        # 16 x 21 x 79
        self.conv_glyph5 = nn.Conv2d(16, 8, 3, padding=1)

        # 4 x 9 x 9
        self.conv_crop1 = nn.Conv2d(1, 16, 3, padding=1)
        # 16 x 9 x 9
        self.conv_crop2 = nn.Conv2d(16, 16, 3, padding=1)
        # 16 x 9 x 9
        self.conv_crop3 = nn.Conv2d(16, 16, 3, padding=1)
        # 16 x 9 x 9
        self.conv_crop4 = nn.Conv2d(16, 16, 3, padding=1)
        # 16 x 9 x 9
        self.conv_crop5 = nn.Conv2d(16, 8, 3, padding=1)

        self.linear1 = nn.Linear(8 * 21 * 79 + 8 * 9 * 9, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, action_space.n)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # forward pass, using relu activations
        glyphs = torch.from_numpy(np.stack([s['glyphs'] for s in x])).float().unsqueeze(1).to(
                self.device) / 5991.0
        crop = torch.from_numpy(np.stack([s['glyphs_crop'] for s in x])).float().unsqueeze(1).to(
                self.device) / 5991.0

        # 5 conv layers for glyphs
        glyphs = F.relu(self.conv_glyph1(glyphs))
        glyphs = F.relu(self.conv_glyph2(glyphs))
        glyphs = F.relu(self.conv_glyph3(glyphs))
        glyphs = F.relu(self.conv_glyph4(glyphs))
        glyphs = F.relu(self.conv_glyph5(glyphs))

        # 5 conv layers for the crop
        crop = F.relu(self.conv_crop1(crop))
        crop = F.relu(self.conv_crop2(crop))
        crop = F.relu(self.conv_crop3(crop))
        crop = F.relu(self.conv_crop4(crop))
        crop = F.relu(self.conv_crop5(crop))

        # flatten from the multidimensional feature map from the conv layers, to a single dimension for the fully-connected layers
        glyphs = glyphs.view(-1, 8 * 21 * 79)
        crop = crop.view(-1, 8 * 9 * 9)

        # concatenate the two flattened feature maps
        x = torch.cat((glyphs, crop), 1)

        # 2 fully-connected layers
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)  # no activation on the output layer
        return x
