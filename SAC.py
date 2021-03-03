
import os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import math

from train import ws, agent, nn_base
from train.bu import OO
from train.nn_base import MLP, weight_init

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, cfg):
        super().__init__()

        self.log_std_bounds = cfg.log_std_bounds
        self.trunk = MLP(cfg.env_size.obs_dim, [cfg.hidden_dim]*cfg.hidden_depth, 2 * cfg.env_size.action_dim)

        self.outputs = dict()
        self.apply(weight_init)
        self.double()

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        # for i, m in enumerate(self.trunk):
        #     if type(m) == nn.Linear:
        #         logger.log_param(f'train_actor/fc{i}', m, step)


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, cfg):
        super().__init__()

        self.Q1 = MLP(cfg.env_size.obs_dim + cfg.env_size.action_dim, [cfg.hidden_dim]*cfg.hidden_depth, 1)
        self.Q2 = MLP(cfg.env_size.obs_dim + cfg.env_size.action_dim, [cfg.hidden_dim]*cfg.hidden_depth, 1)

        self.double()
        
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        #assert len(self.Q1) == len(self.Q2)
        # for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
        #     assert type(m1) == type(m2)
        #     if type(m1) is nn.Linear:
        #         logger.log_param(f'train_critic/q1_fc{i}', m1, step)
        #         logger.log_param(f'train_critic/q2_fc{i}', m2, step)

class LogAlpha(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class SACAgent(agent.Agent):
    def __init__(self, rootcfg, progress, meter, replay_buffer):
        super(SACAgent, self).__init__(rootcfg, progress, meter, replay_buffer)

    def build_model(self):
        acfg = self.rootcfg.agent

        self.rootcfg.update(env=OO(
            size=OO(
                obs_dim=self.rootcfg.env.state_line_dim,
                action_dim=self.rootcfg.env.action_dim,
                action_range=[self.rootcfg.env.action_min, self.rootcfg.env.action_max],
            )))

        self.critic = DoubleQCritic(acfg.critic_cfg).to(self.cfg.device)
        self.critic_target = DoubleQCritic(acfg.critic_cfg).to(self.cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(acfg.actor_cfg).to(self.cfg.device)

        self.log_alpha = torch.tensor(np.log(acfg.init_temperature)).to(self.cfg.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -acfg.action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=acfg.actor_lr,
                                                betas=acfg.actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=acfg.critic_lr,
                                                 betas=acfg.critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=acfg.alpha_lr,
                                                    betas=acfg.alpha_betas)

        self.modules.update(
            actor = self.actor,
            critic = self.critic,
            critic_target = self.critic_target,
        )
        self.optimizers.update(
            actor = self.actor_optimizer,
            critic = self.critic_optimizer,
            log_alpha_optimizer = self.log_alpha_optimizer,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def action(self, pre_state, state, exploration_rate_epsilon:float=0):
        obs = torch.DoubleTensor(np.array(state, np.float64)).to(self.cfg.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if exploration_rate_epsilon>0 else dist.mean
        action = action.clamp(self.cfg.action_min, self.cfg.action_max)
        assert action.ndim == 2 and action.shape[0] == 1
        action = nn_base.to_np(action[0])
        return action, action, [] 

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.cfg.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.cfg.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, batch_size):
        idx, memory, ISWeights = self.replay_buffer.sample(batch_size)
        step, pre_state, state, state_digest, next_state, action, action_prob, reward, done = memory
        self.meter.log('train/batch_reward', np.mean(reward).item(), self.progress.global_step)
        
        state = torch.DoubleTensor(state)
        action = torch.DoubleTensor(action)
        reward = torch.DoubleTensor(reward)
        next_state = torch.DoubleTensor(next_state)
        not_done = torch.DoubleTensor(1-np.array(done))

        self.update_critic(state, action, reward, next_state, not_done, self.meter, self.progress.global_step)

        if self.progress.global_step % self.cfg.actor_update_frequency == 0:
            self.update_actor_and_alpha(state, self.meter, self.progress.global_step)

        if self.progress.global_step % self.cfg.critic_target_update_frequency == 0:
            nn_base.soft_update_params(self.critic, self.critic_target,
                                     self.cfg.critic_tau)



if __name__ == '__main__':
    ws.main(base_file=__file__, 
            agent_class=SACAgent,
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
