import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from nn import Actor, Critic
from utils import OUNoise, ReplayBuffer

class DDPG_Agent():
    """Deep Deterministic Policy Gradients (DDPG) Actor-Critic Agent."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed, config):
        """Initialize Agent object.
        
        Params
        ======
            state_size (int): dimension of each state.
            action_size (int): dimension of each action.
            n_agents (int): number of agents (for multi-agent training).
            random_seed (int): random seed.
            config (obj): Config class object with model configuration.
        """
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_agents = n_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, 
                                 config.fc1_act, config.fc2_act, config.bn).to(config.DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed, 
                                  config.fc1_act, config.fc2_act, config.bn).to(config.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, n_agents, random_seed, 
                                   config.fc1_crit, config.fc2_crit, config.fc3_crit, config.bn).to(config.DEVICE)
        self.critic_target = Critic(state_size, action_size, n_agents, random_seed, 
                                    config.fc1_crit, config.fc2_crit, config.fc3_crit, config.bn).to(config.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic, 
                                           weight_decay=config.l2_reg)
        
        # make sure local and target networks start with same weights
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_decay = config.noise_decay

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.config.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.noise_decay = np.maximum(self.noise_decay*self.config.noise_decay_factor, self.config.min_noise)
            action += self.noise_decay * self.noise.sample()
        return np.clip(action, -1, 1)
    
    def target_act(self, state):
        """Returns actions for given state as per target policy."""
        action = self.actor_target(state).cpu().data.numpy()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
        
    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_params, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_params.data.copy_(local_param.data)   
            
class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradients (MADDPG) Actor-Critic Agent."""
    def __init__(self, state_size, action_size, n_agents, config, random_seed=420):
        """Initialize Multi-Agent object.
        
        Params
        ======
            state_size (int): dimension of each state.
            action_size (int): dimension of each action.
            n_agents (int): number of agents (for multi-agent training).
            config (obj): Config class object with model configuration.
            random_seed (int): random seed.
        """
        self.config = config
        self.maddpg_agent = [DDPG_Agent(state_size, action_size, n_agents, random_seed, config) 
                            for _ in range(n_agents)]
        self.n_agents = n_agents
        
        # Replay memory
        self.memory = ReplayBuffer(config.buffer_size, config.batch_size, n_agents, random_seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def tensor(self, np_arr):
        """Numpy -> Pytorch tensor"""
        return torch.from_numpy(np_arr).float().to(self.config.DEVICE)
    
    def reset(self):
        """Reset noise of all agents."""
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()
    
    def get_noise(self):
        """Get noise of all agents."""
        return [ddpg_agent.noise_decay for ddpg_agent in self.maddpg_agent]
    
    def act(self, obs_all_agents, add_noise=True):
        """Get actions from all agents in the MADDPG object."""
        actions = [agent.act(obs, add_noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents):
        """Get target network actions from all the agents in the MADDPG object."""
        target_actions = [self.tensor(ddpg_agent.target_act(obs)) for ddpg_agent, obs in 
                          zip(self.maddpg_agent, obs_all_agents)]
        return target_actions
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.config.update_every

        if len(self.memory) > self.config.batch_size and self.t_step == 0:
            for _ in range(self.config.update_times):
                for a_i, _ in enumerate(self.maddpg_agent):
                    experiences = self.memory.sample()
                    self.learn(experiences, a_i)
                    
    def learn(self, experiences, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples.
            agent_number (int): agent number (index).
        """
        config = self.config
        
        agent = self.maddpg_agent[agent_number]

        states, actions, rewards, next_states, dones = experiences
        all_states = torch.cat(states, dim=1).to(self.config.DEVICE)
        all_next_states = torch.cat(next_states, dim=1).to(self.config.DEVICE)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        all_next_action = self.target_act(next_states)
        all_next_action = torch.cat(all_next_action, dim=1).to(self.config.DEVICE)
        
        with torch.no_grad():
            Q_targets_next = agent.critic_target(all_next_states, all_next_action)

        # Compute Q targets for current states (y_i)
        Q_targets = \
            rewards[agent_number] + (config.gamma * Q_targets_next * (1 - dones[agent_number]))

        # Compute critic loss
        all_actions = torch.cat(actions, dim=1).to(self.config.DEVICE)
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if config.clip_grad_crit:
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 
                                           config.clip_grad_crit) # Clip Gradients
        agent.critic_optimizer.step()
        
        ##########################
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions_pred = [ self.maddpg_agent[i].actor_local(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor_local(ob).detach()
                   for i, ob in enumerate(states) ]
        
        all_actions_pred = torch.cat(all_actions_pred, dim=1)
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()

        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        if config.clip_grad_act:
            torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), 
                                           config.clip_grad_act) # Clip Gradients
        agent.actor_optimizer.step()
        
        #####################
        
        self.soft_update(agent.critic_local, agent.critic_target, config.tau)
        self.soft_update(agent.actor_local, agent.actor_target, config.tau) 

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from).
            target_model: PyTorch model (weights will be copied to).
            tau (float): interpolation parameter.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save_agents(self):
        """Save Weights of every agent"""
        for idx, ddpg_agent in enumerate(self.maddpg_agent):
            torch.save(ddpg_agent.actor_local.state_dict(), f'MADDPG_chkpnt_actor_{idx}.pth')
            torch.save(ddpg_agent.critic_local.state_dict(),f'MADDPG_chkpnt_critic_{idx}.pth')
            torch.save(ddpg_agent.actor_target.state_dict(), f'MADDPG_chkpnt_actor_target_{idx}.pth')
            torch.save(ddpg_agent.critic_target.state_dict(),f'MADDPG_chkpnt_critic_target_{idx}.pth')

    def load_agents(self):
        """Load Weights of every agent"""
        for idx, ddpg_agent in enumerate(self.maddpg_agent):
            ddpg_agent.actor_local.load_state_dict(torch.load(f'MADDPG_chkpnt_actor_{idx}.pth'))
            ddpg_agent.critic_local.load_state_dict(torch.load(f'MADDPG_chkpnt_critic_{idx}.pth'))
            ddpg_agent.actor_target.load_state_dict(torch.load(f'MADDPG_chkpnt_actor_target_{idx}.pth'))
            ddpg_agent.critic_target.load_state_dict(torch.load(f'MADDPG_chkpnt_critic_target_{idx}.pth'))    