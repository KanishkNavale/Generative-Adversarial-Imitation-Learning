from typing import List, Tuple
import os
import json

import gym
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Actor(torch.nn.Module):
    def __init__(self,
                 input_dimension: int,
                 action_dimension: int,
                 density: int = 512,
                 learning_rate: float = 1e-3,
                 name: str = '') -> None:
        super(Actor, self).__init__()

        self.name = name

        self.H1 = torch.nn.Linear(input_dimension, density)
        self.H2 = torch.nn.Linear(density, density)
        self.H3 = torch.nn.Linear(density, density)
        self.H4 = torch.nn.Linear(density, action_dimension)

        self.log_stds = torch.nn.Parameter(torch.zeros(1, action_dimension[0]))

        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def forward(self, state) -> torch.Tensor:

        x = F.relu(self.H1(state))
        x = F.relu(self.H2(x))
        x = F.relu(self.H3(x))
        action_probability = F.softmax(self.H4(x))

        return action_probability

    def pick_action(self, state):
        with torch.no_grad():
            action_probability = self.forward(state)
            action = torch.argmax(action_probability, dim=-1)
            return action.cpu().numpy().item()

    def save_checkpoint(self, path: str = ''):
        torch.save(self.state_dict(), os.path.join(path, self.name + '.pth'))

    def load_checkpoint(self, path: str = ''):
        self.load_state_dict(torch.load(os.path.join(path, self.name + '.pth')))


class PPO():
    def __init__(self,
                 env: gym.Env,
                 gamma: float = 0.99) -> None:
        self.env = env
        self.state_dims: int = env.observation_space.shape[0]
        self.action_dims: int = env.action_space.n

        self.optimization_steps: int = 0
        self.gamma = 0.99
