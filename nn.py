import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, bn=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            seed (int): Random seed.
            fc1_units (int): Number of nodes in first hidden layer.
            fc2_units (int): Number of nodes in second hidden layer.
            bn (bool): Do batch normalization.
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn = bn
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model weights."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps an agent state -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = self.fc1(state)
        if self.bn:
            x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        if self.bn:
            x = self.bn2(x)
        x = F.leaky_relu(x)
        
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, n_agents, seed, fc1_units=256, fc2_units=128, fc3_units=128, bn=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            n_agents (int): number of agents (for multi-agent training).
            seed (int): Random seed.
            fcs1_units (int): Number of nodes in the first hidden layer.
            fc2_units (int): Number of nodes in the second hidden layer.
            fc3_units (int): Number of nodes in the third hidden layer.
            bn (bool): Do batch normalization.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.n_agents = n_agents
        self.fc1 = nn.Linear(state_size*n_agents, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size*n_agents, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.bn = bn
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model weights."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs for all agents -> Q-values."""
        
        x = F.leaky_relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        
        x = self.fc2(x)
        if self.bn:
            x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x = self.fc3(x)
        if self.bn:
            x = self.bn3(x)
        x = F.leaky_relu(x)
        
        return torch.sigmoid(self.fc4(x))