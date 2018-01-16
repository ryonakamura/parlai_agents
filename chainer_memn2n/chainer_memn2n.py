# ParlAI MemN2NAgent by Chainer
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
from parlai.parlai_agents.save.save import SaveAgent

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
import copy
import os
import random
import pprint
import pickle


class MemN2NAgent(Agent, chainer.Chain):

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
            opt['cuda'] = not opt['no_cuda'] and chainer.cuda.available
            global xp
            if opt['cuda']:
                print('[ Using CUDA ]')
                cuda.get_device(opt['gpu']).use()
                xp = cuda.cupy
            else:
                xp = np

            # dictionary (null: 0, end: 1, unk: 2, start: 3)
            self.id = 'MemN2N'
            self.dict = DictionaryAgent(opt)
            self.sos = self.dict.start_token
            self.sos_lt = xp.array(self.dict.txt2vec(self.sos)).astype(xp.int32)
            self.eos = self.dict.end_token
            self.eos_lt = xp.array(self.dict.txt2vec(self.eos)).astype(xp.int32)
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

            init = chainer.initializers.Normal(scale=0.05)
            # mean is 0 and standard deviation is scale
            # init = chainer.initializers.Uniform(scale=0.05)
            # low is -scale and high is scale
            
            # params
            super(Agent, self).__init__()
            self.embeddings = []
            self.temp_encs = []

            # No weight tying
            if self.weight_tying == 'Nothing':
                mat_num = nl*2 # [A1, C1, A2, C2, ..., Ak, Ck]

            # Adjacent weight tying
            elif self.weight_tying == 'Adjacent':
                mat_num = nl+1 # [(A1), (C1, A2), (C2, A3), ..., (Ck-1, Ak), (Ck)]

            # Layer-wise (RNN-like) weight tying
            elif self.weight_tying == 'Layer-wise':
                mat_num = 2 # [(A1, A2, ..., Ak), (C1, C2, ..., Ck)]
                super(Agent, self).add_link('H', L.Linear(hs, hs, initialW=init))

            for i in range(1, mat_num+1):
                # E* is used for embedding matrix A and C.
                self.embeddings += [('E%d' % i, L.EmbedID(vs, hs, initialW=init))]
                # T* is used for Temporal Encoding.
                self.temp_encs += [('T%d' % i, L.EmbedID(ms, hs, initialW=init))]

            for embed, temp in zip(self.embeddings, self.temp_encs):
                super(Agent, self).add_link(*embed)
                super(Agent, self).add_link(*temp)

            # No weight tying
            if self.weight_tying == 'Nothing':
                super(Agent, self).add_link('B', L.EmbedID(vs, hs, initialW=init))
                super(Agent, self).add_link('W', L.EmbedID(vs, hs, initialW=init))

            # Adjacent and Layer-wise weight tying
            if self.weight_tying in ['Adjacent', 'Layer-wise']:
                # Question sentence embedding matrix B shares weight with
                # memory embedding matrix E1 in the first layer.
                self.B = self.embeddings[0][1]

                # Matrix W in the projection layer shares weight with
                # memory embedding matrix Ek in the last layer.
                self.W = self.embeddings[-1][1]

            # Doubles the hidden layer to generate two words.
            super(Agent, self).add_link('double', L.Linear(hs, hs*2, initialW=init))

            # debug
            if True:
                pp = pprint.PrettyPrinter(indent=4)
                print("\n\nparam:")
                pp.pprint([l for l in super(Agent, self).namedlinks()]) # namedparams
                w = self.B.W.data
                print("param 'B':\n", w)
                print("max:", np.max(w))
                print("min:", np.min(w))
                print("mean:", np.mean(w))
                print("var:", np.var(w))
                print("hist:", np.histogram(w, bins=[-1,-0.5,0,0.5,1]))
                print("hist:", np.histogram(w, bins=[-0.1,-0.05,0,0.05,0.1]))

            # optims
            if self.optimizer_type == 'SGD':
                self.optimizer = chainer.optimizers.SGD(lr=self.learning_rate)
            elif self.optimizer_type == 'AdaGrad':
                self.optimizer = chainer.optimizers.AdaGrad(lr=self.learning_rate)
            elif self.optimizer_type == 'Adam':
                self.optimizer = chainer.optimizers.Adam(alpha=self.learning_rate)
            self.optimizer.setup(self)
            # self.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

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

    def _position_encoding(self, xs):
        # Making l for Position Encoding
        if xs.data.ndim == 2: # if xs is question
            bs, ss = xs.data.shape[:]
            xs = xs.reshape((bs, 1, ss))
        bs, ms, ss = xs.data.shape[:]
        hs = self.hidden_size
        # Make k
        k = xp.arange(1, hs + 1, dtype=xp.float32)
        k = k.reshape((1, hs)) # 1 x hidden
        # Make l
        l = xp.ones((bs, ms, ss, hs)).astype(xp.float32)
        for _x, x in enumerate(xs.data):
            for _s, s in enumerate(x):
                # Make J
                J = xp.count_nonzero(s)
                if J != 0:
                    # Make j
                    j = xp.arange(1, J + 1, dtype=xp.float32)
                    j = j.reshape((J, 1)) # non-0 sequence x 1
                    # Make l
                    _l = (1 - j / J) - (k / hs) * (1 - 2 * j / J)
                    l[_x, _s, :J, :] = _l
        return l

    def _embedding(self, embed, x, l=None): # batch x memory x sequence
        # If x is question, memory size rank is not exist
        e = embed(x) # batch x memory x sequence x hidden
        
        # Position Encoding
        if self.use_position_encoding:
            if e.data.ndim == 3: # if x is question
                bs, ss, hs = e.data.shape[:]
                e = e.reshape((bs, 1, ss, hs))
            e = l * e # batch x memory x sequence x hidden
            if e.data.shape[1] == 1: # if x is question
                e = e.reshape((bs, ss, hs))

        # With negative axis, the same rank of statements or question be specified.
        e = F.sum(e, axis=-2) # batch x memory x hidden
        return e

    def _attention(self, u, x, l, i): # batch x hidden
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

        m = self._embedding(A, x, l) # m: batch x memory x hidden
        c = self._embedding(C, x, l)

        # Temporal Encoding
        if self.use_temporal_encoding:
            bs, ms = m.data.shape[:2]
            # e.g. np.arange(4 - 1, -1, -1) => [3 2 1 0]
            inds = xp.arange(ms - 1, -1, -1, dtype=xp.int32)
            tm = TA(inds) # tm: memory x hidden
            tc = TC(inds)
            tm = F.broadcast_to(tm, (bs,) + tm.data.shape) # batch x memory x hidden
            tc = F.broadcast_to(tc, (bs,) + tc.data.shape)
            m = m + tm
            c = c + tc
        
        c = F.swapaxes(c, 2, 1) # c: batch x hidden x memory
        p = F.batch_matmul(m, u) # p: batch x memory x 1

        # Linear Start
        if not self.use_linear_start:
            p = F.softmax(p)

        o = F.batch_matmul(c, p) # o: batch x hidden x 1
        o = o[:, :, 0] # o: batch x hidden

        # Layer-wise (RNN-like) weight tying
        if self.weight_tying == 'Layer-wise':
            u = self.H(u)

        u = o + u

        # debug
        if False:
            print("a:",[round(i[0],4) for i in p[0].data.tolist()])
        
        if not self.save_atte_done:
            self.attn_weight.append(F.squeeze(p, axis=2).data)
        return u # batch x hidden

    def _forward(self, x, q):
        bs = len(x.data)
        nl = self.num_layers
        l = None

        if self.use_position_encoding:
            l = self._position_encoding(q)
        u = self._embedding(self.B, q, l)

        if self.use_position_encoding:
            l = self._position_encoding(x)
        for i in range(nl):
            u = self._attention(u, x, l, i)
        
        u = self.double(u)
        us = F.split_axis(u, 2, axis=1)
        xs = [F.linear(u, self.W.W) for u in us] # xs: [batch x vocab, batch x vocab]
        # xs = [F.softmax(x) for x in xs]

        preds = [[] for _ in range(bs)]
        ids = [F.argmax(x, axis=1) for x in xs] # ids: [batch x 1, batch x 1]
        for i in range(2):
            for j in range(bs):
                token = self.vec2txt([ids[i][j].data])
                preds[j].append(token)

        return xs, preds

    def train(self, x, q, y):
        with chainer.using_config('train', True):
            self.zero_grad()
            xs, preds = self._forward(x, q)
            loss = 0
            y = F.transpose(y, axes=(1, 0)) # y: 2 x batch
            for i in range(2):
                loss += F.softmax_cross_entropy(xs[i], y[i])
            loss.backward()
            self.update_params()
            self.saver.loss(float(loss.data))

        # debug
        if False:
            print("loss:",round(float(loss.data),4))
        return preds

    def predict(self, x, q):
        with chainer.using_config('train', False):
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

        if len(exs) == 0:
            return (None,)*5
        
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
            if 'labels' in exs[0] and self.use_random_noise:
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
        tensor = xp.zeros((len(xs), x_max_len, arr_max_len)).astype(xp.int32)
        for i, x in enumerate(xs):
            offset = ms - len(x)
            for j, arr in enumerate(x):
                tensor[i, offset + j][:len(arr)] = arr
        x = chainer.Variable(tensor) # batch x sentence (memory size) x word
        
        # debug
        if False:
            print("\n\nx:",[self.vec2txt(tensor[0][i]) for i in range(len(tensor[0]))])
        
        # query
        arr_max_len = max([len(arr) for arr in qs])
        tensor = xp.zeros((len(qs), arr_max_len)).astype(xp.int32)
        for j, arr in enumerate(qs):
            tensor[j][:len(arr)] = arr
        q = chainer.Variable(tensor) # batch x word
        
        # debug
        if False:
            print("q:",self.vec2txt(tensor[0]))

        # label
        y = None
        if 'labels' in exs[0]:
            ys = [self.txt2vec(' '.join(ex['labels']))[:2] for ex in exs]
            tensor = xp.zeros((len(ys), 2)).astype(xp.int32)
            for j, arr in enumerate(ys):
                tensor[j][:len(arr)] = arr
            y = chainer.Variable(tensor) # batch x 2
        
        # debug
        if False:
            print("y:",self.vec2txt(tensor[0]))

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

        if x is None:
            return batch_reply

        if y is not None:
            preds = self.train(x, q, y) # [['bedroom', '__NULL__'], ...]
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
        # x, q, y and elements of self.attn_weight are all numpy.ndarray
        attn_weight = self.attn_weight
        if self.use_cuda:
            attn_weight = [cuda.to_cpu(a) for a in attn_weight]
        attn_weight = xp.array(attn_weight).transpose(1,2,0) # batch x source x layer
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
        path = self.path if path is None else path
        chainer.serializers.save_npz(path, self)
        self.saver.loss(save=True)
        with open(path + '.opt', 'wb') as write:
            pickle.dump(self.opt, write)

    def load(self, path):
        chainer.serializers.load_npz(path, self)

    def get_opt(self, path):
        with open(path + '.opt', 'rb') as read:
            opt = pickle.load(read)
            return opt["options"]
