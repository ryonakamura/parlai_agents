# ParlAI MemN2NAgent by PyTorch
# 
# Auther: Ryo Nakamura @ Master's student at NAIST in Japan
# Date: 2018/01/10
# Contact: @_Ryobot on Twitter (faster than other methods)
#          nakamura.ryo.nm8[at]is.naist.jp
# Project: https://github.com/ryonakamura/parlai_agents
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.parlai_agents.save.save import SaveAgent

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import copy
import os
import random
import pprint
import json
import pickle


class MemN2NAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        SaveAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('MemN2N Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=64,
            help='size of the hidden layers and embeddings')
        agent.add_argument('-ms', '--memorysize', type=int, default=50,
            help='size of the memory (both key and value)')
        agent.add_argument('-nl', '--numlayers', type=int, default=3,
            help='number of memory layers (hops)')
        agent.add_argument('-wt', '--weighttying', type=str, default='Adjacent',
            help='select weight tying from Adjacent, Layer-wise, Nothing')
        agent.add_argument('-pe', '--positionencoding', type='bool', default=True,
            help='if True, use a Position Encoding for word embedding')
        agent.add_argument('-te', '--temporalencoding', type='bool', default=True,
            help='if True, use a Temporal Encoding for sentence memorization')
        agent.add_argument('-rn', '--randomnoise', type='bool', default=True,
            help='if True, use a Random Noise to regularize TE')
        agent.add_argument('-ls', '--linearstart', type='bool', default=True,
            help='if True, use a Linear Start (remove softmax for the memory layers)')
        agent.add_argument('-opt', '--optimizer', type=str, default='Adam',
            help='select optimizer from SGD, AdaGrad, Adam')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.001,
            help='learning rate')
        agent.add_argument('--no-cuda', action='store_true', default=False,
            help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=0,
            help='which GPU device to use')

    def __init__(self, opt, shared=None):
        super(MemN2NAgent, self).__init__(opt, shared)
        if not shared:
            # don't enter this loop for shared instantiations            
            if not "babi" in opt['task']:
                raise NotImplementedError('Other than bAbI Task not supported yet '
                                          'but I will implement full version soon.')
            # option
            self.opt = opt
            # self.opt = self.get_opt(opt['model_file'])

            # cuda
            opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
            if opt['cuda']:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            # dictionary (null: 0, end: 1, unk: 2, start: 3)
            self.id = 'MemN2N'
            self.dict = DictionaryAgent(opt)
            self.sos = self.dict.start_token
            self.sos_lt = torch.LongTensor(self.dict.txt2vec(self.sos))
            self.eos = self.dict.end_token
            self.eos_lt = torch.LongTensor(self.dict.txt2vec(self.eos))
            self.null = self.dict.null_token

            # model settings
            self.hidden_size = opt['hiddensize']
            self.memory_size = opt['memorysize']
            self.num_layers = opt['numlayers']
            self.weight_tying = opt['weighttying']
            self.use_position_encoding = opt['positionencoding']
            self.use_temporal_encoding = opt['temporalencoding']
            self.use_random_noise = opt['randomnoise']
            self.use_linear_start = opt['linearstart']
            self.optimizer_type = opt['optimizer']
            self.learning_rate = opt['learningrate']
            self.path = opt.get('model_file', None)
            vs = len(self.dict)
            hs = self.hidden_size
            ms = self.memory_size
            nl = self.num_layers
            
            # params
            self.params = {}

            # No weight tying
            if self.weight_tying == 'Nothing':
                mat_num = nl*2 # [A1, C1, A2, C2, ..., Ak, Ck]

            # Adjacent weight tying
            elif self.weight_tying == 'Adjacent':
                mat_num = nl+1 # [(A1), (C1, A2), (C2, A3), ..., (Ck-1, Ak), (Ck)]

            # Layer-wise (RNN-like) weight tying
            elif self.weight_tying == 'Layer-wise':
                mat_num = 2 # [(A1, A2, ..., Ak), (C1, C2, ..., Ck)]
                self.H = nn.Linear(hs, hs)
                self.params['H'] = self.H

            self.embeddings = []
            self.temp_encs = []

            for i in range(1, mat_num+1):
                # E* is used for embedding matrix A and C.
                self.embeddings += [('E%d' % i, nn.Embedding(vs, hs,
                            padding_idx=0, scale_grad_by_freq=True))]
                # T* is used for Temporal Encoding.
                self.temp_encs += [('T%d' % i, nn.Embedding(ms, hs,
                            padding_idx=0, scale_grad_by_freq=True))]

            for e, t in zip(self.embeddings, self.temp_encs):
                self.params[e[0]] = e[1]
                self.params[t[0]] = t[1]

            # No weight tying
            if self.weight_tying == 'Nothing':
                self.B = nn.Embedding(vs, hs,
                                padding_idx=0, scale_grad_by_freq=True)
                self.W = nn.Linear(hs, vs)

            # Adjacent and Layer-wise weight tying
            if self.weight_tying in ['Adjacent', 'Layer-wise']:
                # Question sentence embedding matrix B shares weight with
                # memory embedding matrix E1 in the first layer.
                self.B = self.embeddings[0][1]

                # Matrix W in the projection layer shares weight with
                # memory embedding matrix Ek in the last layer.
                self.W = nn.Linear(hs, vs)
                self.W.weight = self.embeddings[-1][1].weight
                # FYI: https://discuss.pytorch.org/t/
                # how-to-create-model-with-sharing-weight/398

            self.params['B'] = self.B
            self.params['W'] = self.W

            # Doubles the hidden layer to generate two words.
            self.double = nn.Linear(hs, hs*2)
            self.params['double'] = self.double

            # Initialize the weights
            for var in self.params.values():
                nn.init.normal(var.weight.data, mean=0, std=0.05)
                # nn.init.uniform(var.weight.data, a=-0.05, b=0.05)
                # nn.init.xavier_normal(var.weight.data)

            # debug
            if True:
                pp = pprint.PrettyPrinter(indent=4)
                print("\n\nparam:")
                pp.pprint(self.params)
                w = self.params['B'].weight.data
                print("param 'B':", w)
                print("max:", torch.max(w))
                print("min:", torch.min(w))
                print("mean:", torch.mean(w))
                print("var:", torch.var(w))
                print("hist:", np.histogram(w.numpy(), bins=[-1,-0.5,0,0.5,1]))
                print("hist:", np.histogram(w.numpy(), bins=[-0.1,-0.05,0,0.05,0.1]))

            # funcs
            self.softmax = nn.Softmax()
            self.log_softmax = nn.LogSoftmax()
            self.nll_loss = nn.NLLLoss()

            self.funcs = [self.softmax, self.log_softmax, self.nll_loss]

            # optims
            self.optims = {}
            lr = self.learning_rate
            for name, var in self.params.items():
                # torch.nn.utils.clip_grad_norm(var.parameters(), 1)
                if self.optimizer_type == 'SGD':
                    self.optims[name] = optim.SGD(var.parameters(), lr=lr)
                elif self.optimizer_type == 'AdaGrad':
                    self.optims[name] = optim.Adagrad(var.parameters(), lr=lr)
                elif self.optimizer_type == 'Adam':
                    self.optims[name] = optim.Adam(var.parameters(), lr=lr)
                else:
                    raise ValueError("""An invalid option for `-opt` was supplied,
                                    options are ['SGD', 'AdaGrad', 'Adam']""")

            # others
            self.saver = SaveAgent(opt)
            self.attn_weight = []
            self.save_atte_done = False
            self.observation = {}
            self.use_cuda = opt.get('cuda', False)
            self.path = opt.get('model_file', None)
            if self.use_cuda:
                self.cuda()
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
                self.load(opt['model_file'])
        
        self.episode_done = True

    def cuda(self):
        for var in self.params.values():
            var.cuda()
        for var in self.funcs:
            var.cuda()

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

    def _position_encoding(self, xs):
        # Making l for Position Encoding
        if xs.dim() == 2: # if xs is question
            bs, ss = xs.size()
            xs = xs.view(bs, 1, ss)
        bs, ms, ss = xs.size()
        hs = self.hidden_size
        # Make k
        k = torch.arange(1, hs + 1)
        k = k.view(1, hs) # 1 x hidden
        # Make pe
        pe = torch.ones(bs, ms, ss, hs)
        for _x, x in enumerate(xs):
            for _s, s in enumerate(x):
                # Make J
                try:
                    J = torch.nonzero(s.data).size(0)
                    # Make j
                    j = torch.arange(1, J + 1)
                    j = j.view(J, 1) # non-0 sequence x 1
                    # Make l
                    l = (1 - j / J) - (k / hs) * (1 - 2 * j / J)
                    pe[_x, _s, :J, :] = l
                except:
                    pass
        if self.use_cuda:
            pe = pe.cuda(async=True)
        pe = Variable(pe)
        return pe

    def _embedding(self, embed, x, pe=None): # batch x memory x sequence
        # If x is question, memory size rank is not exist
        _x = x
        if _x.dim() == 3: # if x is statements
            bs, ms, ss = x.size()
            x = x.view(bs*ms, ss)
        e = embed(x) # batch x memory x sequence x hidden
        if _x.dim() == 3: # if x is statements
            e = e.view(bs, ms, ss, -1)

        # Position Encoding
        if self.use_position_encoding:
            if e.dim() == 3: # if x is question
                bs, ss, hs = e.size()
                e = e.view(bs, 1, ss, hs)
            e = pe * e # batch x memory x sequence x hidden
            if e.size(1) == 1: # if x is question
                e = e.view(bs, ss, hs)

        # With negative dim, the same rank of statements or question be specified.
        e = e.sum(dim=-2) # batch x memory x hidden
        return e

    def _attention(self, u, x, i, pe=None): # u: batch x hidden
        # A and C are used for embedding matrix.
        # TA and TC are used for Temporal Encoding.

        # No weight tying
        if self.weight_tying == 'Nothing':
            A = self.embeddings[i*2][1]
            C = self.embeddings[i*2+1][1]
            TA = self.temp_encs[i*2][1]
            TC = self.temp_encs[i*2+1][1]

        # Adjacent weight tying
        elif self.weight_tying == 'Adjacent':
            A = self.embeddings[i][1]
            C = self.embeddings[i+1][1]
            TA = self.temp_encs[i][1]
            TC = self.temp_encs[i+1][1]
        
        # Layer-wise weight tying
        elif self.weight_tying == 'Layer-wise':
            A = self.embeddings[0][1]
            C = self.embeddings[1][1]
            TA = self.temp_encs[0][1]
            TC = self.temp_encs[1][1]  

        m = self._embedding(A, x, pe) # m: batch x memory x hidden
        c = self._embedding(C, x, pe)

        # Temporal Encoding
        if self.use_temporal_encoding:
            bs, ms, _ = m.size()
            # e.g. np.arange(4 - 1, -1, -1) => [3 2 1 0]
            inds = torch.arange(ms - 1, -1, -1, out=torch.LongTensor())
            if self.use_cuda:
                inds = inds.cuda(async=True)
            inds = Variable(inds)
            tm = TA(inds) # tm: memory x hidden
            tc = TC(inds)
            m = m + tm.expand_as(m) # batch x memory x hidden
            c = c + tm.expand_as(c)
        
        c = c.permute(0,2,1)
        p = torch.bmm(m, u.unsqueeze(2)) # p: batch x memory x 1

        # Linear Start
        if not self.use_linear_start:
            p = p.squeeze(2)
            p = self.softmax(p)
            p = p.unsqueeze(2)

        o = torch.bmm(c, p) # o: batch x hidden x 1
        o = o[:, :, 0] # o: batch x hidden

        # Layer-wise (RNN-like) weight tying
        if self.weight_tying == 'Layer-wise':
            u = self.H(u)

        u = o + u

        # debug
        if False:
            print("a:",[round(i[0],4) for i in p[0].data.tolist()])

        if not self.save_atte_done:
            self.attn_weight.append(p.squeeze(2).data)
        return u # batch x hidden

    def _forward(self, x, q, drop=False):
        bs = x.size(0)
        nl = self.num_layers
        pe = None

        if self.use_position_encoding:
            pe = self._position_encoding(q)
        u = self._embedding(self.B, q, pe)

        if self.use_position_encoding:
            pe = self._position_encoding(x)
        for i in range(nl):
            u = self._attention(u, x, i, pe)
        
        u = self.double(u)
        us = u.chunk(2, dim=1)
        xs = [self.log_softmax(self.W(u)) for u in us] # xs: [*(batch x vocab,)*2]

        preds = [[] for _ in range(bs)]
        ids = [x.max(dim=1)[1] for x in xs] # ids: [batch x 1, batch x 1]
        for i in range(2):
            for j in range(bs):
                token = self.vec2txt(ids[i][j].data.tolist())
                preds[j].append(token)

        return xs, preds

    def train(self, x, q, y):
        self.train_step = True
        xs, preds = self._forward(x, q, drop=True)
        loss = 0
        y = y.transpose(0, 1) # y: 2 x batch
        for i in range(2):
            loss += self.nll_loss(xs[i], y[i])
        self.zero_grad()
        loss.backward()
        self.update_params()
        self.saver.loss(float(loss.data[0]))

        # debug
        if False:
            print("loss:",round(float(loss.data),4))
        return preds

    def predict(self, x, q):
        self.train_step = False
        _, preds = self._forward(x, q)
        if random.random() < 0.1:
            print('prediction:', preds[0])
        return preds

    def batchify(self, obs):
        """Convert batch observations `text` and `label` to
        rank 3 tensor `x` and rank 2 tensor `q`, `y`
        """
        exs = [ex for ex in obs if 'text' in ex]
        ids = [ex['id'] for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        if 'labels' in exs[0]:
            self.train_step = True
        else:
            self.train_step = False
        
        # input
        ms = self.memory_size
        xs = [ex['text'].split('\n') for ex in exs]
        # Get last question sentence
        qs = [self.txt2vec(x.pop()) for x in xs]
        # Remove question sentences and randomly add 10% of empty sentences
        parsed_xs = []
        for x in xs:
            x_mask = ['?' not in s for s in x]
            x = [s for s, b in zip(x, x_mask) if b]

            # Random Noise
            if self.train_step and self.use_random_noise:
                parsed_x = []
                for s in x:
                    parsed_x.append(s)
                    if random.random() < 0.1:
                        parsed_x.append('')
                x = parsed_x

            parsed_xs.append(x[-ms:])
        xs = parsed_xs
        # Make variable
        xs = [[self.txt2vec(sent) for sent in x] for x in xs]
        x_max_len = ms # max([len(x) for x in xs])
        arr_max_len = max(max(len(arr) for arr in x) for x in xs)
        lt = torch.LongTensor(len(xs), x_max_len, arr_max_len).fill_(0)
        for i, x in enumerate(xs):
            offset = ms - len(x)
            for j, arr in enumerate(x):
                if len(arr) == 0:
                    continue
                lt[i, offset + j][:len(arr)] = arr
        if self.use_cuda:
            lt = lt.cuda(async=True)
        x = Variable(lt) # batch x sentence (memory size) x word
        
        # debug
        if False:
            print("\n\nx:",[self.vec2txt(lt[0][i]) for i in range(len(lt[0]))])
        
        # query
        arr_max_len = max([len(arr) for arr in qs])
        lt = torch.LongTensor(len(qs), x_max_len).fill_(0)
        for j, arr in enumerate(qs):
            lt[j][:len(arr)] = arr
        if self.use_cuda:
            lt = lt.cuda(async=True)
        q = Variable(lt) # batch x word
        
        # debug
        if False:
            print("q:",self.vec2txt(lt[0]))

        # label
        y = None
        if 'labels' in exs[0]:
            ys = [self.txt2vec(' '.join(ex['labels']))[:2] for ex in exs]
            lt = torch.LongTensor(len(ys), 2).fill_(0)
            for j, arr in enumerate(ys):
                lt[j][:len(arr)] = arr
            if self.use_cuda:
                lt = lt.cuda(async=True)
            y = Variable(lt) # batch x 2
        
        # debug
        if False:
            print("y:",self.vec2txt(lt[0]))

        return x, q, y, ids, valid_inds

    def batch_act(self, observations):
        # observations:
        #       [{'label_candidates': {'office', ...},
        #       'episode_done': False, 'text': 'Daniel ... \nWhere is Mary?',
        #       'labels': ('office',), 'id': 'babi:Task10k:1'}, ...]
        # In joint training, observation examples are randomly selected from all tasks
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        x, q, y, ids, valid_inds = self.batchify(observations)

        if len(x) == 0:
            return batch_reply

        if y is not None:
            preds = self.train(x, q, y) # # [['bedroom', '__NULL__'], ...]
            if not self.save_atte_done:
                self.save_attention(x.data, q.data, y.data, ids)
        else:
            preds = self.predict(x, q)

        for i in range(len(preds)):
            batch_reply[valid_inds[i]]['text'] = ' '.join(c for c in preds[i]
                            if c != self.eos and c != self.null)
        # debug
        if False:
            print("preds:",preds[0])

        return batch_reply # [{'text': 'bedroom', 'id': 'MemN2N'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]

    def save_attention(self, x, q, y, ids):
        # x, q, y and elements of attn_weight are all torch.FloatTensor
        attn_weight = self.attn_weight
        attn_weight = [a.unsqueeze(2) for a in attn_weight]
        attn_weight = torch.cat(attn_weight, dim=2) # batch x source x layer
        if self.use_cuda:
            attn_weight = attn_weight.cpu()
        attn_weights = attn_weight.tolist()
        
        sources = [[self.vec2txt([c for c in s if c != 0]) for s in ex] for ex in x]
        targets = [[self.vec2txt([c]) if c != 0 else '' for c in ex] for ex in y]
        others = [self.vec2txt([c for c in ex if c != 0]) for ex in q]

        for i in range(len(x)):
            id, w, s, t, o = ids[i], attn_weights[i], sources[i], targets[i], others[i]
            self.save_atte_done = self.saver.attention(id, w, s, t, o)
            if self.save_atte_done:
                print('[ Save Attention Done ]')
                break
        self.attn_weight = []

    def save(self, path=None):
        model = {}
        model['opt'] = self.opt
        for name, var in self.params.items():
            model[name] = var.state_dict()
        path = self.path if path is None else path
        with open(path, 'wb') as write:
            torch.save(model, write)
        self.saver.loss(save=True)
        with open(path + '.opt', 'wb') as write:
            pickle.dump(self.opt, write)

    def load(self, path):
        with open(path, 'rb') as read:
            model = torch.load(read)
        for name, var in self.params.items():
            var.load_state_dict(model[name])

    def get_opt(self, path):
        with open(path + '.opt', 'rb') as read:
            opt = pickle.load(read)
            return model['opt']
