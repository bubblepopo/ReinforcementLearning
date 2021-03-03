from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import csv
import shutil
import torch
import numpy as np
from termcolor import colored
import time
import re

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int', 0),
    ('step', 'S', 'int', 0),
    ('episode_reward', 'R', 'float', 0) 
]

AGENT_TRAIN_FORMAT = {
    'sac': [
        ('episode', 'E', 'int', 0),
        ('step', 'S', 'int', 0),
        ('episode_reward', 'ER', 'float', 0),
        ('running_reward', 'RR', 'float', 0),
        ('epsilon', 'EP', 'float', 0),
        ('actor_loss', 'PL', 'float', 0),
        ('critic_loss', 'QL', 'float', 0),
        ('critic_2_loss', 'Q2', 'float', 0),
        ('alpha_loss', 'AL', 'float', 0),
        ('actor_entropy', 'AE', 'float', 0),
        ('duration', 'D', 'time', 0),
    ]
}

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, cfg, file_name, formating):
        self.cfg = cfg
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_write_time = time.time()
        self.data = []

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = {}
        for f in self._formating:
            data[f[0]] = f[3]
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        self.data += [data]
        if time.time() > self._csv_write_time + self.cfg.save_exceed_seconds:
            with open(self._csv_file_name, 'w') as csv_file:
                csv_writer = csv.DictWriter(csv_file,
                                            fieldnames=data.keys(),
                                            restval=0.0,
                                            extrasaction='ignore')
                csv_writer.writeheader()
                csv_writer.writerows(self.data)
            self._csv_write_time = time.time()
    
    def _float_format(self, value):
        if abs(value) >= 1e7:
            s = f'{value:1.3e}'
            s = re.sub(r'e(.)0', lambda m: 'e'+m.group(1), s)
            while len(s) > 8 and s.index('.') > 0:
                s = re.sub(r'(\.\d*)\de', lambda m: m.group(1)+'e', s).replace(".e","e")
        elif abs(value) <= 1e-2:
            s = f'{value:1.3e}'
            s = re.sub(r'e(.)0', lambda m: 'e'+m.group(1), s)
            while len(s) > 8 and s.index('.') > 0:
                s = re.sub(r'(\.\d*)\de', lambda m: m.group(1)+'e', s).replace(".e","e")
        else:
            s = f'{value:.06f}'
            if len(s) > 8:
                s = s[:8]
        if len(s) < 8:
            s = s.ljust(8, ' ')
        return s

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            s = self._float_format(value)
            return f'{key}: {s}'
        elif ty == 'time':
            return f'{key}: {value:04.1f}s'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'{prefix: <14}']
        for key, disp_key, ty, dv in self._formating:
            value = data.get(key, dv)
            pieces.append(self._format(disp_key, value, ty))
        print(' '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Meter(object):
    def __init__(self,
                 cfg,
                 log_dir,
                 save_tb=False,
                 log_frequency=10000,
                 agent='sac'):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        if save_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except:
                    print("logger.py warning: Unable to remove tb directory")
                    pass
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        # each agent has specific output format for training
        assert agent in AGENT_TRAIN_FORMAT
        self.train_mg = MetersGroup(cfg, os.path.join(log_dir, 'train'),
                                     formating=AGENT_TRAIN_FORMAT[agent])
        self.eval_mg = MetersGroup(cfg, os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT)

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1, log_frequency=1):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self.train_mg if key.startswith('train') else self.eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self.train_mg.dump(step, 'train', save)
            self.eval_mg.dump(step, 'eval', save)
        elif ty == 'eval':
            self.eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self.train_mg.dump(step, 'train', save)
        else:
            raise f'invalid log type: {ty}'
