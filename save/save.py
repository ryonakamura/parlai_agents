# ParlAI SaveAgent (Save losses and attention weights)
# 
# Auther: Ryo Nakamura @ Master's student at NAIST in Japan
# Date: 2017/10/18
# Contact: @_Ryobot on Twitter (faster than other methods)
#          nakamura.ryo.nm8[at]is.naist.jp
# Project: https://github.com/ryonakamura/parlai_agents
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.utils import Timer

import os
import json


class SaveAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        save = argparser.add_argument_group('Save Arguments')
        save.add_argument('-sltim', '--save-loss-every-n-secs', type=int, default=10,
            help='second interval to save losses')
        save.add_argument('-sae', '--save-attention-exs', type=int, default=5,
            help='number of examples to save attention weights')
        return save

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            # don't enter this loop for shared instantiations
            self.path = opt['model_file']
            
            # loss
            self.loss_time = Timer()
            self.losses = []
            self.save_loss_every_n_secs = opt['save_loss_every_n_secs']
            
            # attention
            if opt['task'] == "babi:All1k":
                self.tasks = {'babi:Task1k:'+str(i):0 for i in range(1,21)}
            elif opt['task'] == "babi:All10k":
                self.tasks = {'babi:Task10k:'+str(i):0 for i in range(1,21)}
            else:
                self.tasks = {task:0 for task in opt['task'].split(',')}
            self.attention_weights = {task:[] for task in self.tasks.keys()}
            self.save_attention_exs = opt['save_attention_exs']

    def loss(self, loss=None, save=False):
        # if `save` is True, save losses to a file.
        if loss:
            if self.save_loss_every_n_secs >= 0 and \
                            self.loss_time.time() > self.save_loss_every_n_secs:
                self.losses.append(loss)
                self.loss_time.reset()
        elif save:
            print('[ Save losses ]')
            path = self.path + '.loss'
            if os.path.isfile(path):
                with open(path, 'r') as read:
                    dictionary = json.load(read)
                    self.losses = dictionary["losses"] + self.losses
            directory = {"losses": self.losses}
            with open(path, 'w') as write:
                json.dump(directory, write, indent=4)
            self.losses = []

    def attention(self, task_id, weight, source, target, other=None):
        done = False
        if task_id in self.tasks.keys():
            if self.tasks[task_id] < self.save_attention_exs:
                self.tasks[task_id] += 1
                self.attention_weights[task_id].append({
                                "weight": weight,
                                "source": source,
                                "target": target,
                                "other": other
                                })

        if all([i >= self.save_attention_exs for i in self.tasks.values()]):
            done = True
            print('[ Save attention weights ]')
            path = self.path + '.atte'
            directory = {"attention_weights": self.attention_weights}
            with open(path, 'w') as write:
                json.dump(directory, write, indent=4)
        return done
