
import os, sys
if __name__ != '__main__': sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from bu import OO, Tracer, Config
Tracer.trace("Loading dependency libraries ...")

import argparse
import random, time
from datetime import datetime, timedelta

import numpy as np
import gym
import torch

from meter import Meter
from agent import Agent
from replay_buffer import ReplayBuffer
from video_recorder import VideoRecorder


def cfg(**key_value_pairs):
    '''
        1. 命令行参数会合并覆盖cfg函数调用参数，统称输入参数
        2. 输入参数会覆盖默认参数和配置文件中定义的参数
        3. 程序运行时参数优先级最高，会合并覆盖前面所有参数
    '''
    env_name=["CartPole-v1", "Pendulum-v0", "Breakout-ram-v4", ][0]  # OpenAI gym environment name
    default_args = OO("default_args", 
            env_name=env_name,
            base_file=__file__,
            agent_class=Agent,
            base_dir="${os.path.dirname(os.path.realpath(base_file))}",
            base_name="${os.path.basename(os.path.realpath(base_file)).replace('.py','')}",
            config_file="${base_dir}/cfg.${env_name}.yaml",
            workdir=os.path.expanduser('~/data/drl.${agent_class.__name__}.${env_name}/'),
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=0,
            num_eval_epochs=1,
            num_train_epochs=1e9,
            reward_forward=0.9,
            replay_buffer_capacity=1000,
            batch_train_episodes=1,
            policy_delay=1,
            train_stop_reward=500,
            train_max_frame=-1,
            eval_max_frame=1000,
            num_seed_steps=0,
            batch_size=128, # LT num_seed_steps
            save_exceed_seconds=10,
            save_prompt_message=False,
            save_video_exceed_reward=0,
            eval_expect_top_reward_percent=0.8,
            running_idle_rate=0.02,
            exploration_rate_init=0,
            exploration_rate_decay=0.9,
            exploration_rate_min=0,
            agent=OO(
                gamma=0.9, # reward discount
                tau=0.01,  # soft replacement
                batch_size="${batch_size}",
            ).__dict__,
        )
    parser = argparse.ArgumentParser()
    for k,v in default_args.items():
        parser.add_argument("--"+k, default=None, help="Default: "+str(v))
    cmd_run_args = OO("cmd_run_args")
    for k,v in parser.parse_args().__dict__.items():
        if v is not None:
            cmd_run_args.update({k:v})
    input_args = OO("input_args")
    input_args.update(key_value_pairs)
    input_args.update(cmd_run_args)
    # 根据输入参数调整动态参数
    default_args.update(input_args,
            # 刷新包含依赖变量的默认值
            base_dir="${os.path.dirname(os.path.realpath(base_file))}",
            base_name="${os.path.basename(os.path.realpath(base_file)).replace('.py','')}",
            config_file="${base_dir}/${base_name}.${env_name}.yaml",
            workdir=os.path.expanduser('~/data/drl.${base_name}.${env_name}/'),
            env=OO(name='${env_name}',
                render=True).__dict__,
            agent=OO(
                batch_size="${batch_size}",
            ).__dict__,
        )
    # 覆盖高优先设置
    default_args.update(input_args)
    Tracer.trace("Parsing config file: %s"%(default_args.config_file))
    return Config(default_args.config_file, default=default_args, override=input_args)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

class Workspace():
    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.exists(cfg.workdir): os.makedirs(cfg.workdir)
        Tracer.FILE = cfg.workdir+"log_"+datetime.now().strftime("%Y%m%d%H%M%S")+".txt"
        self.progress = OO(epoch = 1, exploration_noise = 1, num_train_iteration = 0, num_critic_update_iteration = 0, num_actor_update_iteration = 0)
        self.progress.update(episode=1, 
                episode_step=0, 
                global_step=0, 
                env_top_reward=0, 
                evaluate_reward=0, 
                eval_top_reward=-9e99, 
                train_running_reward=0,
                exploration_rate_epsilon=1)

        self.workdir = cfg.workdir
        self.meter = Meter(cfg, cfg.workdir)
        self.env = gym.make(cfg.env.name)
        env = self.env

        # 命令行参数、函数调用参数和reload配置文件中定义的参数，都会被这里定义的值重新覆盖
        cfg.override(
                device=torch.device(cfg.device),
                agent=OO(
                    device="${device}",
                ).__dict__,
            )
        cfg.override(env=OO(
                spec = env.spec,
                reward_range = env.reward_range,
                state_space_dim = env.observation_space.shape,
                state_line_dim = np.prod(env.observation_space.shape),
                state_digest_dim = 128,
                # 离散 1 或连续 0
                action_discrete = 1 if isinstance(env.action_space, gym.spaces.Discrete) else 0,
                # 动作维度
                action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0],
                # 动作取值下界
                action_min = 0 if isinstance(env.action_space, gym.spaces.Discrete) else float(env.action_space.low[0]),
                # 动作取值上界
                action_max = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else float(env.action_space.high[0]),
            ))
        cfg.override(agent=OO(
                state_space_dim = cfg.env.state_space_dim,
                state_line_dim = cfg.env.state_line_dim,
                state_digest_dim = cfg.env.state_digest_dim,
                action_discrete = cfg.env.action_discrete,
                action_dim = cfg.env.action_dim,
                action_min = cfg.env.action_min,
                action_max = cfg.env.action_max,
                actor = OO(
                    lr = 1e-2,
                    betas = [0.9, 0.999],
                ),
                critic = OO(
                    lr = 1e-2,
                    betas = [0.9, 0.999],
                ),
            ))
        
        assert self.cfg.exploration_rate_init >= 0 and self.cfg.exploration_rate_init < 1
        Tracer.trace(self.cfg)
        
        self.video_recorder = VideoRecorder()
        self.replay_buffer = ReplayBuffer(int(cfg.replay_buffer_capacity))
        self.agent = cfg.agent_class(cfg, self.progress, self.meter, self.replay_buffer)

        self.temp_buffer = []

        self.load()
        self.last_save_time = time.time()
        self.last_rest_time = time.time()
        self.last_log_time = time.time()
        self.last_meter_time = time.time()
        
        print("-"*150)

    def save(self, best_model=False):
        def safe_torch_save(obj, filename):
            torch.save(obj, filename+'.tmp')
            if os.access(filename, os.F_OK): os.remove(filename)
            os.rename(filename+'.tmp', filename)
        agent_model = {}
        for (name, mod) in self.agent.modules.items():
            agent_model[name] = mod.state_dict()
        safe_torch_save(agent_model, self.cfg.workdir+'/training_model.drl')
        if best_model: safe_torch_save(agent_model, self.cfg.workdir+'/agent_model.drl')
        progress = {
                'progress': self.progress.__dict__,
                'meters.train': self.meter.train_mg.data,
                'meters.eval': self.meter.eval_mg.data,
            }
        for (name, optim) in self.agent.optimizers.items():
            progress[name] = optim.state_dict()
        safe_torch_save(progress, self.cfg.workdir+'/progress.drl')
        safe_torch_save({
                'memory': self.replay_buffer.__dict__,
            }, self.cfg.workdir+'/replay_buffer.drl')
        if 'save_prompt_message' in self.cfg and self.cfg.save_prompt_message:
            Tracer.trace("Model has been saved...")
        
    def load(self):
        def safe_torch_load(filename):
            if os.path.exists(filename):
                return torch.load(filename)
            elif os.path.exists(filename+'.tmp'):
                return torch.load(filename+'.tmp')
            else:
                return None
        o = safe_torch_load(self.cfg.workdir+'/agent_model.drl')
        if o is not None:
            for (name, mod) in self.agent.modules.items():
                if name in o: mod.load_state_dict(o[name])
        o = safe_torch_load(self.cfg.workdir+'/progress.drl')
        if o is not None:
            for (name, optim) in self.agent.optimizers.items():
                if name in o: optim.load_state_dict(o[name])
        if o is not None and 'progress' in o: self.progress.__dict__.update(o['progress'])
        if o is not None and 'meters.train' in o: self.meter.train_mg.data = o['meters.train']
        if o is not None and 'meters.eval' in o: self.meter.eval_mg.data = o['meters.eval']
        o = safe_torch_load(self.cfg.workdir+'/replay_buffer.drl')
        if o is not None and 'memory' in o: self.replay_buffer.__dict__.update(o['memory'])
        Tracer.trace("model has been loaded...")

    def evaluate(self):
        self.video_recorder.init()
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_epochs):
            state = self.env.reset()
            pre_state = state
            self.agent.reset()
            done = False
            episode_reward = 0
            step = 0
            while not done and step<self.cfg.eval_max_frame:
                with eval_mode(self.agent):
                    action, action_prob = self.agent.action(pre_state, state, 0)
                prestate = state
                state, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                step += 1
            work_time = time.time() - self.last_rest_time
            time.sleep((work_time*self.cfg.running_idle_rate) if work_time>0 else 0.01)
            self.last_rest_time = time.time()
            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        self.meter.log('eval/episode_reward', average_episode_reward, self.progress.global_step)
        self.meter.dump(self.progress.global_step)
        return average_episode_reward

    def push_replay_buffer(self, info):
        if self.cfg.reward_forward == 0:
            self.replay_buffer.push(info)
        else:
            self.temp_buffer += [info]
            if info[-1][0] != 0:
                for i in reversed(range(len(self.temp_buffer)-1)):
                    if self.temp_buffer[i][-1][0] == 0:
                        self.temp_buffer[i][-2][0] += self.cfg.reward_forward * self.temp_buffer[i+1][-2][0]
                    else:
                        self.temp_buffer[i][-2][0] += 0
                for i in range(len(self.temp_buffer)):
                    self.replay_buffer.push(self.temp_buffer[i])
                self.temp_buffer.clear()

    def train_episode(self, train_max_frame, exploration_rate_epsilon):
        progress = self.progress
        state = self.env.reset()
        pre_state = state
        self.agent.reset()
        done_or_stop = False
        episode_reward = 0
        episode_step = 0
        self.meter.log('train/episode', progress.episode, progress.global_step)
        while not done_or_stop:
            if self.cfg.env.render:
                self.env.render()
            # sample action
            action, action_prob, state_digest = self.agent.action(pre_state, state, exploration_rate_epsilon)
            # one step
            next_state, reward, done, info = self.env.step(action)
            #有时候env输出的observation_space可能与定义的不一样
            #因为action可以是多个连续动作
            assert next_state.shape == self.env.observation_space.shape
            episode_reward += reward
            # if cfg.env_name == "Breakout-ram-v4":
            #     s = next_state * (state != next_state)
            #     s[90] = 0
            #     if np.sum(s) == 0:
            #         done = True
            done_or_stop = done or train_max_frame > 0 and episode_step == train_max_frame
            self.push_replay_buffer(([str(progress.global_step)], pre_state, state, state_digest, next_state, action, action_prob, [reward], [float(done_or_stop)]))
            # run training update
            if self.replay_buffer.data_count >= self.cfg.batch_size:
                for i in range(self.cfg.batch_train_episodes):
                    self.agent.update(self.cfg.batch_size)
                    self.progress.num_train_iteration += 1
                    work_time = time.time() - self.last_rest_time
                    time.sleep((work_time*self.cfg.running_idle_rate) if work_time>0 else 0.01)
                    self.last_rest_time = time.time()
                    if time.time() < self.last_save_time or time.time() - self.last_save_time >= self.cfg.save_exceed_seconds:
                        self.save()
                        self.last_save_time = time.time()
            
            pre_state = state
            state = next_state
            episode_step += 1
            progress.global_step += 1
        return episode_reward

    def train(self):
        def set_random_seed(seed):
            if seed is not None:
                random.seed(seed)
                self.env.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
        # trainning
        set_random_seed(self.cfg.seed)
        self.agent.train()
        progress = self.progress
        while (self.cfg.num_train_epochs < 0 or progress.episode <= self.cfg.num_train_epochs) \
                and (self.cfg.train_stop_reward < 0 or progress.train_running_reward < self.cfg.train_stop_reward) \
                and progress.train_running_reward < progress.env_top_reward:
            self.cfg.reload()
            train_max_frame = self.cfg.train_max_frame
            if isinstance(self.env.spec.max_episode_steps, int) and self.env.spec.max_episode_steps < train_max_frame:
                train_max_frame = self.env.spec.max_episode_steps
            
            if self.progress.global_step < self.cfg.num_seed_steps:
                exploration_rate_epsilon = 1
            elif self.progress.exploration_rate_epsilon == 1:
                exploration_rate_epsilon = self.progress.exploration_rate_epsilon = self.cfg.exploration_rate_init
            else:
                exploration_rate_epsilon = self.progress.exploration_rate_epsilon

            start_time = time.time()
            episode_reward = self.train_episode(train_max_frame, exploration_rate_epsilon)
            if progress.episode > 1:
                progress.train_running_reward = 0.05 * episode_reward + (1 - 0.05) * progress.train_running_reward
            else:
                progress.train_running_reward = episode_reward

            self.meter.log('train/epsilon', exploration_rate_epsilon, progress.global_step)
            self.meter.log('train/episode_reward', episode_reward, progress.global_step)
            self.meter.log('train/running_reward', progress.train_running_reward, progress.global_step)
            self.meter.log('train/duration', time.time() - start_time, progress.global_step)
            self.meter.dump(progress.global_step)

            if self.progress.exploration_rate_epsilon < 1:
                if self.progress.exploration_rate_epsilon > self.cfg.exploration_rate_min:
                    self.progress.exploration_rate_epsilon *= self.cfg.exploration_rate_decay
                if self.progress.exploration_rate_epsilon < self.cfg.exploration_rate_min:
                    self.progress.exploration_rate_epsilon = self.cfg.exploration_rate_min

            if progress.train_running_reward > progress.env_top_reward * self.cfg.eval_expect_top_reward_percent:
                self.meter.log('eval/episode', progress.episode, progress.global_step)
                self.progress.evaluate_reward = self.evaluate()
                if self.progress.evaluate_reward > self.progress.eval_top_reward:
                    self.save(best_model=True)
                    if self.progress.evaluate_reward > self.cfg.save_video_exceed_reward:
                        vrfile = f'{self.workdir}/eval_top_reward.mp4'
                        self.video_recorder.save(vrfile)
                        self.progress.eval_top_reward = self.progress.evaluate_reward
            
            work_time = time.time() - self.last_rest_time
            time.sleep((work_time*self.cfg.running_idle_rate) if work_time>0 else 0.01)
            self.last_rest_time = time.time()
            
            progress.episode += 1
            # Finished One Episode
        return progress.train_running_reward

    def run(self):
        while True:
            if isinstance(self.env.spec.reward_threshold, float) and self.env.spec.reward_threshold>self.progress.env_top_reward:
                self.progress.env_top_reward = self.env.spec.reward_threshold 
            elif self.cfg.train_stop_reward > self.progress.env_top_reward:
                self.progress.env_top_reward = self.cfg.train_stop_reward
            elif self.progress.evaluate_reward > self.progress.env_top_reward:
                self.progress.env_top_reward = self.progress.evaluate_reward
            if self.progress.train_running_reward < self.progress.env_top_reward:
                self.progress.train_running_reward = self.train()
            else:
                self.progress.evaluate_reward = self.evaluate()


def main(**args):
    try:
        # for warning when env.render(): ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to ...
        os.system("defaults write org.python.python ApplePersistenceIgnoreState NO")
        Workspace(cfg(**args)).run()
    except KeyboardInterrupt:
        pass
    finally:
        os.system("defaults write org.python.python ApplePersistenceIgnoreState YES")

if __name__ == '__main__':
    main()
    