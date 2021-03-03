import os

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from bu import OO, Tracer
import nn_base

class Agent(object):
    """ 默认实现 Deep Q-Network """
    training = False
    modules = OO()
    optimizers = OO()
    eps = np.finfo(np.float32).eps.item()

    def __init__(self, rootcfg, progress, meter, replay_buffer):
        self.rootcfg = rootcfg
        self.progress = progress
        self.meter = meter
        self.replay_buffer = replay_buffer

        self.cfg = rootcfg.agent
        self.build_model()

    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""

    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""
        self.training = training
        for (name, mod) in self.modules.items():
            if isinstance(mod, list) or isinstance(mod, tuple):
                for m in mod:
                    m.train(training)
            else:
                mod.train(training)

    def build_model(self):
        self.modules.update(
            actor=nn_base.MLP(self.cfg.state_space_dim, [512], self.cfg.action_dim),
        )
        self.optimizers.update(
            actor = torch.optim.Adam(self.modules.actor.parameters(), lr=self.cfg.actor.lr, betas=self.cfg.actor.betas),
        )
        self.actor = self.modules.actor

    def action(self, pre_state, state, exploration_rate_epsilon:float=0):
        pre_state = torch.unsqueeze(nn_base.to_tensor(pre_state), 0)
        state = torch.unsqueeze(nn_base.to_tensor(state), 0)
        a = self.actor(state)
        if self.cfg.action_discrete == 0:
            a = (1 + torch.tanh(a))/2 * (self.cfg.action_max - self.cfg.action_min) + self.cfg.action_min
        else:
            a = torch.softmax(a, dim=1)
        action_prob = a
        if self.cfg.action_discrete == 0: # 连续动作空间
            if exploration_rate_epsilon > 0:
                noise = torch.normal(mean=0, std=exploration_rate_epsilon, size=action_prob.size()) # 包含从给定参数means,std的离散正态分布中抽取随机数
                noise = noise.clamp(-exploration_rate_epsilon, exploration_rate_epsilon)
                action_prob *= (1 + noise)
                action_prob = action_prob.clamp(self.cfg.action_min, self.cfg.action_max)
            action = action_prob.detach().numpy()[0]
        else: # 离散动作空间
            if exploration_rate_epsilon > 0 and np.random.random() < exploration_rate_epsilon:
                action_prob = np.random.random([self.cfg.action_max-self.cfg.action_min])
                action_prob = torch.from_numpy(action_prob)
                action_prob = F.softmax(action_prob, dim=1)
            m = Categorical(action_prob)
            action = m.sample()
            action_prob = m.log_prob(action)
            action = action.item()
        return action, action_prob, []

    def update(self, batch_size):
        """Main function of the agent that performs learning."""
        idx, memory, isweights = self.replay_buffer.sample(-1)
        step, pre_state, state, state_digest, next_state, action, action_prob, reward, done = memory
        self.replay_buffer.remove(idx)
        
        self.meter.log('train/batch_reward', np.mean(reward).item(), self.progress.global_step)

        policy_loss = []
        returns = torch.tensor(reward)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(action_prob, returns):
            policy_loss.append(-log_prob * R)
        self.optimizers.actor.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizers.actor.step()
        self.meter.log('train/actor_loss', policy_loss.item(), self.progress.global_step)


if __name__ == "__main__":
    import main
    main.main()
