import os
os.environ["TORCH_DISABLE_ONEDNN"] = "1"
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GRAD_CLIP   = 10.0
RNN_SIZE    = 512
GOAL_REPR_SIZE = 12


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.pool(x)


class ACNet(nn.Module):
    """
    Actor-Critic network for PRIMAL2.
    Input:
        obs      : (B, NUM_CHANNEL, OBS_SIZE, OBS_SIZE)
        goal_pos : (B, 3)   [dx, dy, mag]
        hx, cx   : (1, B, RNN_SIZE)  LSTM hidden states
    Output:
        policy   : (B, a_size)   softmax probabilities
        value    : (B, 1)
        valids   : (B, a_size)   sigmoid (blocking validity)
        hx, cx   : updated LSTM states
    """

    def __init__(self, num_channel, obs_size, a_size):
        super().__init__()
        self.a_size   = a_size
        self.rnn_size = RNN_SIZE

        # VGG feature extractor
        self.vgg1 = VGGBlock(num_channel, RNN_SIZE // 4)
        self.vgg2 = VGGBlock(RNN_SIZE // 4, RNN_SIZE // 4)

        # after two 2x2 max-pools on OBS_SIZE=11: floor(floor(11/2)/2) = 2
        # conv3: VALID 2x2 on 2x2 → 1x1
        self.conv3 = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE - GOAL_REPR_SIZE, 2)

        # goal embedding
        self.goal_fc = nn.Linear(3, GOAL_REPR_SIZE)

        # residual MLP
        self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)

        # LSTM
        self.lstm = nn.LSTMCell(RNN_SIZE, RNN_SIZE)

        # output heads
        self.policy_head = nn.Linear(RNN_SIZE, a_size)
        self.value_head  = nn.Linear(RNN_SIZE, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # policy head: small init
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        # value head: unit init
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs, goal_pos, hx, cx):
        """
        obs      : (B, C, H, W)
        goal_pos : (B, 3)
        hx, cx   : (B, RNN_SIZE)
        """
        # CNN
        x = self.vgg1(obs)
        x = self.vgg2(x)
        x = self.conv3(x)           # (B, RNN_SIZE - GOAL_REPR_SIZE, 1, 1)
        x = F.relu(x.flatten(1))    # (B, RNN_SIZE - GOAL_REPR_SIZE)

        # goal
        g = F.relu(self.goal_fc(goal_pos))   # (B, GOAL_REPR_SIZE)

        # concat → residual MLP
        hidden = torch.cat([x, g], dim=1)    # (B, RNN_SIZE)
        h1 = F.relu(self.fc1(hidden))
        h2 = self.fc2(h1)
        h3 = F.relu(h2 + hidden)             # residual

        # LSTM
        hx, cx = self.lstm(h3, (hx, cx))

        # heads
        policy_logits = self.policy_head(hx)
        policy  = F.softmax(policy_logits, dim=-1)
        valids  = torch.sigmoid(policy_logits)
        value   = self.value_head(hx)

        return policy, value, valids, hx, cx

    def get_init_hidden(self, batch_size=1, device='cpu'):
        return (torch.zeros(batch_size, self.rnn_size, device=device),
                torch.zeros(batch_size, self.rnn_size, device=device))
