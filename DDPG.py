
from train import ws, agent, nn_base

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn

class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        self.fc1 = nn.Linear(s_dim,30)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,a_dim)
        self.out.weight.data.normal_(0,0.1) # initialization
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x*2
        # if self.action_discrete == 0:
        #     a = (1 + torch.tanh(a))/2 * (self.action_max - self.action_min) + self.action_min
        # else:
        #     a = torch.softmax(a, dim=1)
        return actions_value

class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        self.fcs = nn.Linear(s_dim,30)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(a_dim,30)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization
    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = torch.relu(x+y)
        actions_value = self.out(net)
        return actions_value

class DDPGAgent(agent.Agent):
    """ 实现 Deep Deterministic Policy Gradient """
    
    def __init__(self, rootcfg, progress, meter, replay_buffer):
        super(DDPGAgent, self).__init__(rootcfg, progress, meter, replay_buffer)

    def build_model(self):
        self.modules.update(
            Actor_eval = ANet(self.cfg.state_line_dim, self.cfg.action_dim),
            Actor_target = ANet(self.cfg.state_line_dim, self.cfg.action_dim),
            Critic_eval = CNet(self.cfg.state_line_dim, self.cfg.action_dim),
            Critic_target = CNet(self.cfg.state_line_dim, self.cfg.action_dim),
        )
        self.optimizers.update(
            ctrain = torch.optim.Adam(self.modules.Critic_eval.parameters(),lr=self.cfg.actor.lr, betas=self.cfg.actor.betas),
            atrain = torch.optim.Adam(self.modules.Actor_eval.parameters(),lr=self.cfg.actor.lr, betas=self.cfg.actor.betas),
        )
        self.actor = self.modules.Actor_eval
        self.loss_td = nn.MSELoss()

    def action(self, pre_state, state, exploration_rate_epsilon:float=0):
        pre_state = torch.unsqueeze(nn_base.to_tensor(pre_state), 0)
        state = torch.unsqueeze(nn_base.to_tensor(state), 0)
        action_prob = self.actor(state)
        if self.cfg.action_discrete == 0: # 连续动作空间
            if exploration_rate_epsilon > 0:
                noise = torch.normal(mean=0, std=exploration_rate_epsilon, size=action_prob.size()) # 包含从给定参数means,std的离散正态分布中抽取随机数
                noise = noise.clamp(-exploration_rate_epsilon, exploration_rate_epsilon)
                action_prob *= (1 + noise)
                action_prob = action_prob.clamp(self.cfg.action_min, self.cfg.action_max)
            action_prob = action_prob.detach().numpy()[0]
            action = action_prob
        else: # 离散动作空间
            if exploration_rate_epsilon > 0 and np.random.random() < exploration_rate_epsilon:
                action_prob = np.random.random([self.cfg.action_max-self.cfg.action_min])
                action_prob = torch.from_numpy(action_prob)
                action_prob = F.softmax(action_prob, dim=1)
            m = Categorical(action_prob)
            action = m.sample()
            # action_log_prob = m.log_prob(action)
            action = action.item()
        return action, action_prob, []

    def update(self, batch_size):
        """Main function of the agent that performs learning."""

        for x in self.modules.Actor_target.state_dict().keys():
            eval('self.modules.Actor_target.' + x + '.data.mul_((1-self.cfg.tau))')
            eval('self.modules.Actor_target.' + x + '.data.add_(self.cfg.tau*self.modules.Actor_eval.' + x + '.data)')
        for x in self.modules.Critic_target.state_dict().keys():
            eval('self.modules.Critic_target.' + x + '.data.mul_((1-self.cfg.tau))')
            eval('self.modules.Critic_target.' + x + '.data.add_(self.cfg.tau*self.modules.Critic_eval.' + x + '.data)')

        idx, memory, isweights = self.replay_buffer.sample(batch_size)
        step, pre_state, state, state_digest, next_state, action, action_prob, reward, done = memory
        self.meter.log('train/batch_reward', np.mean(reward).item(), self.progress.global_step)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)

        a = self.modules.Actor_eval(state)
        q = self.modules.Critic_eval(state,a)  
        # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q) 
        #print(q)
        #print(loss_a)
        self.optimizers.atrain.zero_grad()
        loss_a.backward()
        self.optimizers.atrain.step()
        self.meter.log('train/critic_loss', loss_a.item(), self.progress.global_step)

        next_action = self.modules.Actor_target(next_state)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_next = self.modules.Critic_target(next_state,next_action)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = reward + self.cfg.gamma * q_next
        #print(q_target)
        q_v = self.modules.Critic_eval(state,action)
        #print(q_v)
        td_error = self.loss_td(q_target,q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.optimizers.ctrain.zero_grad()
        td_error.backward()
        self.optimizers.ctrain.step()

        self.meter.log('train/actor_loss', td_error.item(), self.progress.global_step)

if __name__ == "__main__":
    ws.main(base_file=__file__, 
            agent_class=DDPGAgent,
            env_name="Pendulum-v0",
            batch_size=128,
            num_seed_steps=0,
            replay_buffer_capacity=1000,
            exploration_rate_init=0.1,
            exploration_rate_decay=0.9,
            exploration_rate_min=0,
        )
