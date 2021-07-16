import torch
import random
import numpy as np
import copy

from collections import namedtuple, deque

class Config:
    ''' Model configuration. '''
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Running on: {DEVICE}')
    def __init__(self):
        self.device = self.DEVICE
        self.buffer_size = int(1e5)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 1e-3
        self.lr_critic = 1e-3
        self.lr_actor = 1e-4
        self.l2_reg = 1e-2
        self.clip_grad_act = 1
        self.clip_grad_crit = 1
        self.update_every = 5
        self.update_times = 2
        self.noise_decay = 1
        self.noise_decay_factor = 0.99
        self.min_noise = 0.01
        
        self.fc1_act = 256
        self.fc2_act = 128
        self.fc1_crit = 256
        self.fc2_crit = 128
        self.fc2_crit = 64
        self.bn = False
        
    def merge(self, config_dict=None):
        if config_dict is None:
            pass
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.
        Params
        ======
            seed (int): random seed.
            mu (float): long-running mean.
            theta (float): speed of mean reversion.
            sigma (float): volatility parameter.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
#         self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, n_agents, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer.
            batch_size (int): size of each training batch.
            n_agents (int): number of agents (for multi-agent training).
            seed (int): random seed.
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.seed = random.seed(seed)
        self.n_agents = n_agents
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = [torch.from_numpy(np.vstack([e.state[index] for e in experiences if e is not None])).float().to(Config.DEVICE)
                  for index in range(self.n_agents)]
        actions = [torch.from_numpy(np.vstack([e.action[index] for e in experiences if e is not None])).float().to(Config.DEVICE)
                  for index in range(self.n_agents)]
        rewards = [torch.from_numpy(np.vstack([e.reward[index] for e in experiences if e is not None])).float().to(Config.DEVICE)
                  for index in range(self.n_agents)]
        next_states = [torch.from_numpy(np.vstack([e.next_state[index] for e in experiences if e is not None])).float().to(Config.DEVICE)
                  for index in range(self.n_agents)]
        dones = [torch.from_numpy(\
                  np.vstack([e.done[index] for e in experiences if e is not None]).astype(np.uint8)).float().to(Config.DEVICE)
                  for index in range(self.n_agents)]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)