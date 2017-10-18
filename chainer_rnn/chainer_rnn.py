# ParlAI RNNAgent by Chainer
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
from parlai.core.dict import DictionaryAgent

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import copy
import os
import random


class RNNAgent(Agent, chainer.Chain):

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('RNN Arguments')
        agent.add_argument('-rnn', '--rnntype', type=str, default='GRU',
            help='choose GRU or LSTM')
        agent.add_argument('-hs', '--hiddensize', type=int, default=64,
            help='size of the hidden layers and embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
            help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.5,
            help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.2,
            help='dropout rate')
        agent.add_argument('--no-cuda', action='store_true', default=False,
            help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=0,
            help='which GPU device to use')

    def __init__(self, opt, shared=None):
        super(RNNAgent, self).__init__(opt, shared)
        if not shared:
            # don't enter this loop for shared instantiations
            opt['cuda'] = not opt['no_cuda'] and chainer.cuda.available
            global xp
            if opt['cuda']:
                print('[ Using CUDA ]')
                cuda.get_device(opt['gpu']).use()
                xp = cuda.cupy
            else:
                xp = np

            self.id = 'RNN'
            self.dict = DictionaryAgent(opt)
            self.observation = {}
            self.rnn_type = opt['rnntype']
            self.hidden_size = opt['hiddensize']
            self.num_layers = opt['numlayers']
            self.dropout_rate = opt['dropout']
            self.learning_rate = opt['learningrate']
            self.use_cuda = opt.get('cuda', False)
            self.path = opt.get('model_file', None)
            vs = len(self.dict)
            hs = self.hidden_size   
            nl = self.num_layers
            dr = self.dropout_rate

            super(Agent, self).__init__(
                            embedding = L.EmbedID(vs, hs),
                            projection = L.Linear(hs, vs))
            if self.rnn_type == 'GRU':
                super(Agent, self).add_link('rnn', L.NStepGRU(nl, hs, hs, dr))
            elif self.rnn_type == 'LSTM':
                super(Agent, self).add_link('rnn', L.NStepLSTM(nl, hs, hs, dr))
            self.dropout = F.dropout
            self.softmax = F.softmax
            self.loss = F.softmax_cross_entropy

            self.optimizer = chainer.optimizers.SGD(lr=self.learning_rate)
            self.optimizer.setup(self)
            self.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

            if self.use_cuda:
                self.cuda()
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
                self.load(opt['model_file'])
        
        self.episode_done = True

    def cuda(self):
        self.to_gpu()

    def txt2vec(self, txt):
        return xp.array(self.dict.txt2vec(txt)).astype(xp.int32)

    def vec2txt(self, vec):
        return self.dict.vec2txt(vec)

    def zero_grad(self):
        self.optimizer.target.cleargrads()

    def update_params(self):
        self.optimizer.update()

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        # At this moment `self.episode_done` is the previous example
        if not self.episode_done:
            # If the previous example is not the end of the episode,
            # we need to recall the `text` mentioned in the previous example.
            # At this moment `self.observation` is the previous example.
            prev_dialogue = self.observation['text']
            # Add the previous and current `text` and update current `text`
            observation['text'] = prev_dialogue + '\n' + observation['text']
        # Overwrite with current example
        self.observation = observation
        # The last example of an episode is provided as `{'episode_done': True}`
        self.episode_done = observation['episode_done']
        return observation

    def forward(self, xs, drop=False):
        out = self.embedding(xs) # out: batch x time x hidden
        if drop:
            out = self.dropout(out, ratio=self.dropout_rate)
        out = [out[i] for i in range(len(out.data))] # out: [*(time x hidden,) * batch]
        if self.rnn_type == 'GRU':
            hy, out = self.rnn(hx=None, xs=out) # out: [*(time x hidden,) * batch]
        elif self.rnn_type == 'LSTM':
            hy, cy, out = self.rnn(hx=None, cx=None, xs=out) # Same as above
        out = [o[-1].reshape(1, -1) for o in out] # out: [*(1 x hidden,) * batch]
        out = F.concat(out, axis=0) # out: batch x hidden
        if drop:
            out = self.dropout(out, ratio=self.dropout_rate)
        out = self.projection(out)
        # out = self.softmax(out) # out: batch x vocab

        preds = []
        idx = F.argmax(out, axis=1) # idx: batch x 1
        for i in idx:
            token = self.vec2txt([i.data])
            preds.append(token)
        return out, preds

    def train(self, xs, ys):
        with chainer.using_config('train', True):
            out, preds = self.forward(xs, drop=True)
            y = F.transpose(ys, axes=(1, 0))[0] # y: batch
            loss = self.loss(out, y)
            self.zero_grad()
            loss.backward()
            loss.unchain_backward()
            self.update_params()
        return preds

    def predict(self, xs):
        with chainer.using_config('train', False):
            out, preds = self.forward(xs)
            if random.random() < 0.1:
                print('prediction:', preds[0])
        return preds

    def batchify(self, obs):
        """Convert batch observations `text` and `label` to rank 2 tensor `xs` and `ys`
        """
        def txt2var(txt, use_offset=True):
            vec = [self.txt2vec(t) for t in txt]
            max_len = max([len(v) for v in vec])
            arr = xp.zeros((len(vec), max_len)).astype(xp.int32) # rank 2 tensor
            for i, v in enumerate(vec):
                offset = 0
                if use_offset:
                    offset = max_len - len(v) # Right justified
                for j, idx in enumerate(v):
                    arr[i][j + offset] = idx
            return chainer.Variable(arr) # batch x time

        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]
        
        xs = [ex['text'] for ex in exs]
        xs = txt2var(xs)
        ys = None
        if 'labels' in exs[0]:
            ys = [' '.join(ex['labels']) for ex in exs]
            ys = txt2var(ys, use_offset=False)
        return xs, ys, valid_inds

    def batch_act(self, observations):
        # observations:
        #       [{'label_candidates': {'office', ...},
        #       'episode_done': False, 'text': 'Daniel ... \nWhere is Mary?',
        #       'labels': ('office',), 'id': 'babi:Task10k:1'}, ...]
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        xs, ys, valid_inds = self.batchify(observations)

        if ys is not None:
            preds = self.train(xs, ys) # ['bedroom', ...]
        else:
            preds = self.predict(xs)

        for i in range(len(preds)):
            batch_reply[valid_inds[i]]['text'] = preds[i]

        return batch_reply # [{'text': 'bedroom', 'id': 'RNN'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        path = self.path if path is None else path
        chainer.serializers.save_npz(path, self)

    def load(self, path):
        chainer.serializers.load_npz(path, self)
