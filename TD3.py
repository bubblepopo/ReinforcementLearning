
import os, sys, random, time
from datetime import datetime, timedelta
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

from train import ws, agent, nn_base
from train.bu import OO

class State(nn.Module):
    def __init__(self, state_dim, state_digest_dim=32):
        super(State, self).__init__()

        self.state_dim = state_dim
        self.state_digest_dim = state_digest_dim

        self.fc1 = nn.Linear(state_dim*2, state_dim*8)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        #self.dropout = nn.Dropout(p=dropout)
        # self.fc2 = nn.Linear(state_dim*2, state_digest_dim*4)
        # self.fc3 = nn.Linear(state_digest_dim*4, state_digest_dim)
        self.fc2 = nn.Linear(state_dim*8, state_digest_dim)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        
    def forward(self, pre_state, state):
        x = torch.cat([pre_state, state], 1)
        #x = torch.normal(x)
        x = self.fc1(x)
        # if state_dim > state_digest_dim * 2:
        #     x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.relu(x)
        # x = self.fc3(x)
        x = torch.relu(x)
        return x

class Actor(nn.Module):

    def __init__(self, state, state_dim, action_dim, action_min, action_max, action_discrete):
        super(Actor, self).__init__()
        self.action_discrete = action_discrete

        #self.state = state
        self.state = State(state_dim)
        self.fc2 = nn.Linear(self.state.state_digest_dim, 64)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.fc3 = nn.Linear(64, action_dim)
        self.fc3.weight.data.normal_(0,0.1) # initialization

        self.action_min = action_min
        self.action_max = action_max

    def forward(self, pre_state, state):
        state_digest = self.state(pre_state, state)
        a = F.relu(self.fc2(state_digest))
        a = self.fc3(a)
        if self.action_discrete == 0:
            a = (1 + torch.tanh(a))/2 * (self.action_max - self.action_min) + self.action_min
        else:
            a = torch.softmax(a, dim=1)
        return a, state_digest


class Critic(nn.Module):

    def __init__(self, state, state_dim, action_dim, action_min, action_max, action_discrete):
        super(Critic, self).__init__()

        #self.state = state
        self.state = State(state_dim)
        # self.fc1s = nn.Linear(self.state.state_digest_dim, 32)
        # if isinstance(env.action_space, gym.spaces.Discrete):
        #     self.fc1a = nn.Linear(action_max - action_min, 32)
        # else:
        #     self.fc1a = nn.Linear(action_dim, 32)
        self.fc2 = nn.Linear(self.state.state_digest_dim+action_dim, 64)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.normal_(0,0.1) # initialization

    def forward(self, pre_state, state, action):
        state_digest = self.state(pre_state, state)
        # x = self.fc1s(state_digest)
        # y = self.fc1a(action)
        # q = torch.cat([x, y], 1)
        q = torch.cat([state_digest, action], 1)
        # q = F.relu(q)
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3Agent(agent.Agent):
    def __init__(self, rootcfg, progress, meter, replay_buffer):
        super(TD3Agent, self).__init__(rootcfg, progress, meter, replay_buffer)

    def build_model(self):
        cfg = self.cfg
        self.state           = State(cfg.state_line_dim, cfg.state_digest_dim).to(cfg.device)
        self.modules.update(
            state           = self.state,
            actor           = Actor(self.state, cfg.state_line_dim, cfg.action_dim, cfg.action_min, cfg.action_max, cfg.action_discrete).to(cfg.device),
            actor_target    = Actor(self.state, cfg.state_line_dim, cfg.action_dim, cfg.action_min, cfg.action_max, cfg.action_discrete).to(cfg.device),
            critic_1        = Critic(self.state, cfg.state_line_dim, cfg.action_dim, cfg.action_min, cfg.action_max, cfg.action_discrete).to(cfg.device),
            critic_1_target = Critic(self.state, cfg.state_line_dim, cfg.action_dim, cfg.action_min, cfg.action_max, cfg.action_discrete).to(cfg.device),
            critic_2        = Critic(self.state, cfg.state_line_dim, cfg.action_dim, cfg.action_min, cfg.action_max, cfg.action_discrete).to(cfg.device),
            critic_2_target = Critic(self.state, cfg.state_line_dim, cfg.action_dim, cfg.action_min, cfg.action_max, cfg.action_discrete).to(cfg.device),
        )
        self.optimizers.update(
            actor = optim.Adam(self.modules.actor.parameters(),lr=cfg.lr_actor,betas=cfg.betas),
            critic_1 = optim.Adam(self.modules.critic_1.parameters(),lr=cfg.lr_critic,betas=cfg.betas),
            critic_2 = optim.Adam(self.modules.critic_2.parameters(),lr=cfg.lr_critic,betas=cfg.betas),
        )
        
        self.modules.actor_target.load_state_dict(self.modules.actor.state_dict())
        self.modules.critic_1_target.load_state_dict(self.modules.critic_1.state_dict())
        self.modules.critic_2_target.load_state_dict(self.modules.critic_2.state_dict())
        
        self.actor = self.modules.actor

    def action(self, pre_state, state, exploration_rate_epsilon:float=0):
        pre_state = torch.unsqueeze(nn_base.to_tensor(pre_state), 0)
        state = torch.unsqueeze(nn_base.to_tensor(state), 0)
        action_prob, state_digest = self.actor(pre_state, state)
        if self.cfg.action_discrete == 0: # 连续动作空间
            if exploration_rate_epsilon > 0:
                noise = torch.normal(mean=0, std=exploration_rate_epsilon, size=action_prob.size()) # 包含从给定参数means,std的离散正态分布中抽取随机数
                noise = noise.clamp(-exploration_rate_epsilon, exploration_rate_epsilon)
                action_prob *= (1 + noise)
                action_prob = action_prob.clamp(self.cfg.action_min, self.cfg.action_max)
            action = action_prob
        else: # 离散动作空间
            if exploration_rate_epsilon > 0 and np.random.random() < exploration_rate_epsilon:
                action_prob = torch.rand_like(action_prob)
                action_prob = F.softmax(action_prob, dim=1)
            m = Categorical(action_prob)
            action = m.sample()
            # action_log_prob = m.log_prob(action)
        return action.detach().numpy()[0], action_prob.detach().numpy()[0], state_digest.detach().numpy()[0]

    def update(self, batch_size):
        idx, memory, ISWeights = self.replay_buffer.sample(batch_size)
        step, pre_state, state, state_digest, next_state, action, action_prob, reward, done = memory
        self.meter.log('train/batch_reward', np.mean(reward).item(), self.progress.global_step)

        pre_state = torch.FloatTensor(pre_state)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action_prob = torch.FloatTensor(action_prob)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        ISWeights = torch.FloatTensor(ISWeights)

        next_action, _ = self.modules.actor_target(state, next_state)
        # Compute target Q-value:
        target_Q1 = self.modules.critic_1_target(state, next_state, next_action)
        target_Q2 = self.modules.critic_2_target(state, next_state, next_action)
        #target_Q = torch.min( target_Q1, target_Q2 )
        target_Q = ( target_Q1 + target_Q2 ) / 2
        target_Q = reward + ((1 - done) * self.cfg.gamma * target_Q).detach()

        # Optimize Critic 1:
        # state = state.view(-1, self.cfg.state_line_dim)
        # action_prob = action_prob.view(-1, self.cfg.action_dim)
        current_Q1 = self.modules.critic_1(pre_state, state, action_prob)
        loss_Q1 = torch.mean(ISWeights * torch.square(current_Q1 - target_Q))
        #loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.optimizers.critic_1.zero_grad()
        loss_Q1.backward()
        self.optimizers.critic_1.step()
        self.meter.log('train/critic_loss', loss_Q1.item(), step=self.progress.global_step)

        # Optimize Critic 2:
        current_Q2 = self.modules.critic_2(pre_state, state, action_prob)
        loss_Q2 = torch.mean(ISWeights * torch.square(current_Q2 - target_Q))
        #loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.optimizers.critic_2.zero_grad()
        loss_Q2.backward()
        self.optimizers.critic_2.step()
        self.meter.log('train/critic_2_loss', loss_Q2.item(), step=self.progress.global_step)

        abs_errors1 = torch.sum(torch.abs(current_Q1 - target_Q), dim=1)
        abs_errors2 = torch.sum(torch.abs(current_Q2 - target_Q), dim=1)
        self.replay_buffer.batch_update(idx, torch.maximum(abs_errors1, abs_errors2).detach().numpy())
        # self.memory.batch_update(tree_idx, abs_errors2.detach().numpy())

        # Delayed policy updates:
        if self.progress.num_train_iteration % self.rootcfg.policy_delay == 0:
            # Compute actor loss:
            action, _ = self.modules.actor(pre_state, state)
            # actor_loss = - self.critic_1(pre_state, state, action).mean()#随着更新的进行Q1和Q2两个网络，将会变得越来越像。所以用Q1还是Q2，还是两者都用，对于actor的问题不大。
            actor_loss = - torch.mean( self.modules.critic_1(pre_state, state, action) )

            # Optimize the actor
            self.optimizers.actor.zero_grad()
            actor_loss.backward()
            self.optimizers.actor.step()
            self.meter.log('train/actor_loss', actor_loss.item(), step=self.progress.global_step)
            for param, target_param in zip(self.modules.actor.parameters(), self.modules.actor_target.parameters()):
                target_param.data.copy_(((1- self.cfg.tau) * target_param.data) + self.cfg.tau * param.data)

            for param, target_param in zip(self.modules.critic_1.parameters(), self.modules.critic_1_target.parameters()):
                target_param.data.copy_(((1 - self.cfg.tau) * target_param.data) + self.cfg.tau * param.data)

            for param, target_param in zip(self.modules.critic_2.parameters(), self.modules.critic_2_target.parameters()):
                target_param.data.copy_(((1 - self.cfg.tau) * target_param.data) + self.cfg.tau * param.data)

            self.progress.num_actor_update_iteration += 1
        self.progress.num_critic_update_iteration += 1

if __name__ == '__main__':
    ws.main(base_file=__file__, 
            agent_class=TD3Agent,
            env_name=["Pendulum-v0",][0],
            batch_size=128,
            num_seed_steps=0,
            replay_buffer_capacity=1000,
            batch_train_episodes=4,
            policy_delay=2,
            exploration_rate_init=0.8,
            exploration_rate_decay=0.99,
            exploration_rate_min=0,
            agent=OO(
                state_digest_dim=64,
                lr_actor=1e-1,
                lr_critic=1e-1,
                betas=(0.9,0.999),
            ),
        )
