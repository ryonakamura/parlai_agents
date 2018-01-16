# ParlAI RNNAgent by PyTorch
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

from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch
import copy
import os
import random


class RNNAgent(Agent):

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
        super().__init__(opt, shared)
        if not shared:
            # don't enter this loop for shared instantiations
            opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
            if opt['cuda']:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            self.id = 'RNN'
            self.dict = DictionaryAgent(opt)
            self.observation = {}
            self.rnn_type = opt['rnntype']
            self.hidden_size = opt['hiddensize']
            self.num_layers = opt['numlayers']
            self.learning_rate = opt['learningrate']
            self.use_cuda = opt.get('cuda', False)
            self.path = opt.get('model_file', None)
            vs = len(self.dict)
            hs = self.hidden_size   
            nl = self.num_layers
            dr = opt['dropout']

            self.embedding = nn.Embedding(vs, hs, padding_idx=0,
                            scale_grad_by_freq=True)
            if self.rnn_type == 'GRU':
                self.rnn = nn.GRU(hs, hs, nl, dropout=dr)
            elif self.rnn_type == 'LSTM':
                self.rnn = nn.LSTM(hs, hs, nl, dropout=dr)
            self.dropout = nn.Dropout(dr)
            self.projection = nn.Linear(hs, vs)
            self.softmax = nn.LogSoftmax()
            self.loss = nn.NLLLoss()

            lr = self.learning_rate
            self.optims = {
                'embedding': optim.SGD(self.embedding.parameters(), lr=lr),
                'rnn': optim.SGD(self.rnn.parameters(), lr=lr),
                'projection': optim.SGD(self.projection.parameters(), lr=lr),
            }
            if self.use_cuda:
                self.cuda()
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
                self.load(opt['model_file'])
        
        self.episode_done = True

    def cuda(self):
        self.embedding.cuda()
        self.rnn.cuda()
        self.dropout.cuda()
        self.projection.cuda()
        self.softmax.cuda()
        self.loss.cuda()

    def txt2vec(self, txt):
        return torch.LongTensor(self.dict.txt2vec(txt))

    def vec2txt(self, vec):
        return self.dict.vec2txt(vec)

    def zero_grad(self):
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        for optimizer in self.optims.values():
            optimizer.step()

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

    def init_zeros(self, bs=1):
        h0 = torch.zeros(self.num_layers, bs, self.hidden_size)
        c0 = torch.zeros(self.num_layers, bs, self.hidden_size)
        if self.use_cuda:
            h0 = h0.cuda(async=True)
            c0 = c0.cuda(async=True)
        return Variable(h0), Variable(c0)

    def forward(self, xs, drop=False):
        out = self.embedding(xs)
        out = torch.transpose(out, 0, 1) # out: time x batch x hidden
        if drop:
            out = self.dropout(out)
        h0, c0 = self.init_zeros(len(xs)) # h0, c0: layer x batch x hidden
        if self.rnn_type == 'GRU':
            out, hn = self.rnn(out, h0) # out: time x batch x hidden
        elif self.rnn_type == 'LSTM':
            out, (hn, cn) = self.rnn(out, (h0, c0)) # Same as above
        out = out[-1] # out: batch x hidden
        if drop:
            out = self.dropout(out)
        out = self.projection(out)
        out = self.softmax(out) # out: batch x vocab

        preds = []
        _, idx = out.max(1) # idx: batch x 1
        for i in idx:
            token = self.vec2txt([i.data[0]])
            preds.append(token)
        return out, preds

    def train(self, xs, ys):
        self.rnn.train()
        out, preds = self.forward(xs, drop=True)
        y = ys.select(1, 0) # y: batch
        loss = self.loss(out, y)
        self.zero_grad()
        loss.backward()
        self.update_params()
        return preds

    def predict(self, xs):
        self.rnn.eval()
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
            lt = torch.LongTensor(len(vec), max_len).fill_(0) # 0 filled rank 2 tensor
            for i, v in enumerate(vec):
                offset = 0
                if use_offset:
                    offset = max_len - len(v) # Right justified
                for j, idx in enumerate(v):
                    lt[i][j + offset] = idx
            if self.use_cuda:
                lt = lt.cuda(async=True)
            return Variable(lt) # batch x time

        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        if len(exs) == 0:
            return (None,)*3

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

        if xs is None:
            return batch_reply

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
        model = {}
        model['embedding'] = self.embedding.state_dict()
        model['rnn'] = self.rnn.state_dict()
        model['projection'] = self.projection.state_dict()

        path = self.path if path is None else path
        with open(path, 'wb') as write:
            torch.save(model, write)

    def load(self, path):
        with open(path, 'rb') as read:
            model = torch.load(read)

        self.embedding.load_state_dict(model['embedding'])
        self.rnn.load_state_dict(model['rnn'])        
        self.projection.load_state_dict(model['projection'])
