# ParlAI AttentionAgent by PyTorch
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

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import copy
import os
import random
import pickle


class AttentionAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)
        SaveAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Attention Arguments')
        agent.add_argument('-rnn', '--rnntype', type=str, default='GRU',
            help='choose GRU or LSTM')
        agent.add_argument('-bi', '--bidirectional', type='bool', default=True,
            help='if True, use a bidirectional encoder for the first layer')
        agent.add_argument('-atte', '--attention', type='bool', default=True,
            help='if True, use an Luong\'s attention')
        agent.add_argument('-cont', '--contextonly', type='bool', default=False,
            help='if True, use only a context vector without using decoder query '
                            'for the final output')
        agent.add_argument('-sf', '--scorefunc', type=str, default='general',
            help='select score function for attention from dot, general, concat')
        agent.add_argument('-tf', '--teacherforcing', type=float, default=1.,
            help='teacher forcing rate')
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
        agent.add_argument('--gpu', type=int, default=-1,
            help='which GPU device to use')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            # don't enter this loop for shared instantiations
            # option
            if opt.get('model_file') and os.path.isfile(opt['model_file'] + '.opt'):
                print('Loading existing options from ' + opt['model_file'])
                opt = self.get_opt(opt['model_file'])
            self.opt = opt

            # cuda
            opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
            if opt['cuda']:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            # dictionary (null: 0, end: 1, unk: 2, start: 3)
            self.id = 'Attention'
            self.dict = DictionaryAgent(opt)
            self.sos = self.dict.start_token
            self.sos_lt = torch.LongTensor(self.dict.txt2vec(self.sos))
            self.eos = self.dict.end_token
            self.eos_lt = torch.LongTensor(self.dict.txt2vec(self.eos))
            self.null = self.dict.null_token

            # model settings
            self.rnn_type = opt['rnntype']
            self.use_bi_encoder = opt['bidirectional']
            self.use_attention = opt['attention']
            self.context_only = opt['contextonly']
            self.score_func = opt['scorefunc']
            self.teacher_forcing_rate = opt['teacherforcing']
            self.hidden_size = opt['hiddensize']
            self.num_layers = opt['numlayers']
            self.learning_rate = opt['learningrate']
            self.longest_label = 1
            rnn = self.rnn_type
            vs = len(self.dict)
            hs = self.hidden_size
            nl = self.num_layers
            bi = 1 if self.use_bi_encoder else 0
            self.use_encoder = True if nl-bi >= 1 else False
            dr = opt['dropout']

            # params
            rnn_cls = {'GRU': nn.GRU, 'LSTM': nn.LSTM}
            self.embedding = nn.Embedding(vs, hs, padding_idx=0,
                            scale_grad_by_freq=True)
            self.decoder = rnn_cls[rnn](hs, hs, nl, dropout=dr)
            self.projection = nn.Linear(hs, vs)
            if self.use_bi_encoder:
                self.bi_encoder = rnn_cls[rnn](hs, hs//2, 1, bidirectional=True)
            if self.use_encoder:
                self.encoder = rnn_cls[rnn](hs, hs, nl-bi, dropout=dr)
            if self.use_attention:
                if self.score_func == 'general':
                    self.W_atte = nn.Linear(hs, hs)
                elif self.score_func == 'concat':
                    self.W_atte = nn.Linear(hs * 2, hs)
                    self.v_atte = nn.Linear(hs, 1)
                if not self.context_only:
                    self.W_cont = nn.Linear(hs * 2, hs)

            self.params = {
                'embedding': self.embedding,
                'decoder': self.decoder,
                'projection': self.projection
            }
            if self.use_bi_encoder:
                self.params['bi_encoder'] = self.bi_encoder
            if self.use_encoder:
                self.params['encoder'] = self.encoder
            if self.use_attention:
                if self.score_func in ['general', 'concat']:
                    self.params['W_atte'] = self.W_atte
                if self.score_func == 'concat':
                    self.params['v_atte'] = self.v_atte
                if not self.context_only:
                    self.params['W_cont'] = self.W_cont

            # funcs
            self.dropout = nn.Dropout(dr)
            self.softmax = nn.LogSoftmax()
            self.softmax_atte = nn.Softmax()
            self.tanh_atte = nn.Tanh()
            self.tanh_cont = nn.Tanh()
            self.loss = nn.NLLLoss()
            
            self.funcs = [self.dropout, self.softmax, self.softmax_atte,
                            self.tanh_atte, self.tanh_cont, self.loss]
            
            # optims
            self.optims = {}
            lr = self.learning_rate
            for name, var in self.params.items():
                self.optims[name] = optim.SGD(var.parameters(), lr=lr)

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

    def _zeros_gen(self, bs=1):
        # Encoder hidden and memory cell start with 0 filled tensor
        hs = self.hidden_size
        nl = self.num_layers
        bi = 1 if self.use_bi_encoder else 0

        bi_h0, bi_c0, h0, c0 = None, None, None, None
        if self.use_bi_encoder:
            bi_h0 = torch.zeros(2, bs, hs//2)
            bi_c0 = torch.zeros(2, bs, hs//2)
            if self.use_cuda:
                bi_h0 = bi_h0.cuda(async=True)
                bi_c0 = bi_c0.cuda(async=True)
            bi_h0, bi_c0 = Variable(bi_h0), Variable(bi_c0)
        if self.use_encoder:
            h0 = torch.zeros(nl-bi, bs, hs)
            c0 = torch.zeros(nl-bi, bs, hs)
            if self.use_cuda:
                h0 = h0.cuda(async=True)
                c0 = c0.cuda(async=True)
            h0, c0 = Variable(h0), Variable(c0)
        return bi_h0, bi_c0, h0, c0

    def _encode(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 0, 1) # x: time x batch x hidden
        if self.train_step:
            x = self.dropout(x)
        bi_h0, bi_c0, h0, c0 = self._zeros_gen(x.size(1)) # layer x batch x hidden
        
        def transform(xn):
            split = [t.squeeze(0) for t in torch.chunk(xn, 2, dim=0)]
            xn = torch.cat(split, dim=1).unsqueeze(0)
            return xn
        
        if self.rnn_type == 'GRU':
            hn, cn = None, None
            if self.use_bi_encoder:
                x, hn = self.bi_encoder(x, bi_h0) # x: time x batch x hidden
                hn = transform(hn)
            if self.use_encoder:
                bi_hn = hn
                x, hn = self.encoder(x, h0) # x: time x batch x hidden
            if self.use_bi_encoder and self.use_encoder:
                hn = torch.cat((bi_hn, hn), dim=0)
        
        elif self.rnn_type == 'LSTM':
            hn, cn = None, None
            if self.use_bi_encoder:
                x, (hn, cn) = self.bi_encoder(x, (bi_h0, bi_c0))
                hn = transform(hn)
                cn = transform(cn)
            if self.use_encoder:
                bi_hn, bi_cn = hn, cn
                x, (hn, cn) = self.encoder(x, (h0, c0))
            if self.use_bi_encoder and self.use_encoder:
                hn = torch.cat((bi_hn, hn), dim=0)
                cn = torch.cat((bi_cn, cn), dim=0)

        if self.train_step:
            x = self.dropout(x)
        return x, hn, cn

    def _sos_gen(self, bs=1):
        # Decoder input starts with SOS tensor
        x = self.sos_lt
        if self.use_cuda:
            x = x.cuda(async=True)
        x = Variable(x) # x: 1
        x = self.embedding(x).unsqueeze(1) # x: 1 x 1 x hidden
        return x.expand(1, bs, x.size(2)) # x: 1 x batch x hidden

    def _attention(self, query, memory):
        # (dot, general, concat) score function was used in Luong et al. 2015
        # (query, key, value) representation was used in KV MemN2N and Transformer.
        # query (decoder RNN output): batch x hidden
        # memory (encoder RNN outputs): time x batch x hidden
        value = memory.transpose(0, 1) # value: batch x time x hidden
        key = value.transpose(1, 2) # key: batch x hidden x time 

        def dot(q, k):
            return torch.bmm(q.unsqueeze(1), k)

        def general(q, k):
            return torch.bmm(self.W_atte(q).unsqueeze(1), k)

        def concat(q, k):
            k = k.transpose(1, 2) # batch x time x hidden
            q = q.unsqueeze(1).expand(*k.size()) # batch x time x hidden
            cat = torch.cat((q, k), dim=2) # batch x time x hidden*2
            out = self.W_atte(cat.view(-1, cat.size(2))) # batch*time x hidden
            out = self.v_atte(self.tanh_atte(out)) # batch*time x 1
            return out.view(cat.size(0), 1, cat.size(1)) # batch x 1 x time

        func = {'dot': dot, 'general': general, 'concat': concat}
        score = func[self.score_func](query, key).squeeze(1) # batch x time
        attn_weight = self.softmax_atte(score).unsqueeze(1) # batch x 1 x time
        context = torch.bmm(attn_weight, value).squeeze(1) # batch x hidden
        if self.context_only:
            x = context
        else:
            x = torch.cat((context, query), dim=1) # batch x hidden*2
            x = self.tanh_cont(self.W_cont(x)) # batch x hidden
        if self.train_step and not self.save_atte_done:
            self.attn_weight.append(attn_weight.squeeze(1).data)
        return x

    def _decode_step(self, x, hn, cn, memory):
        # Use encoder outputs for attention memory
        if self.rnn_type == 'GRU':
            x, hn = self.decoder(x, hn) # x: 1 x batch x hidden
        elif self.rnn_type == 'LSTM':
            x, (hn, cn) = self.decoder(x, (hn, cn))
        x = x.squeeze(0) # x: batch x hidden
        if self.use_attention:
            x = self._attention(x, memory)
        if self.train_step:
            x = self.dropout(x)
        x = self.projection(x)
        x = self.softmax(x) # x: batch x vocab
        return x, hn, cn

    def _decode_and_train(self, memory, hn, cn, y):
        # # Update the model based on the labels
        bs = memory.size(1)
        x = self._sos_gen(bs)
        preds = [[] for _ in range(bs)]
        self.longest_label = max(self.longest_label, y.size(1))
        self.attn_weight = []
        loss = 0

        for i in range(y.size(1)):
            x, hn, cn = self._decode_step(x, hn, cn, memory)
            t = y.select(1, i) # t: batch, select(dim, index)
            loss += self.loss(x, t)
            _, x = x.max(1) # x: batch x 1
            x = x.view(-1)
            # Store prediction
            for j in range(bs):
                token = self.vec2txt([x.data[j]])
                preds[j].append(token)
            # Prepare the next input
            if random.random() < self.teacher_forcing_rate:
                x = self.embedding(t).unsqueeze(0) # 1 x batch x hidden
            else:
                x = self.embedding(x).unsqueeze(0) # 1 x batch x hidden

        self.zero_grad()
        loss.backward()
        self.update_params()
        self.saver.loss(loss.data[0])
        return preds

    def _decode_only(self, memory, hn, cn):
        # Just produce predictions without training the model
        bs = memory.size(1)
        x = self._sos_gen(bs)
        preds = [[] for _ in range(bs)]
        done = [False for _ in range(bs)]
        total_done = 0
        max_len = 0

        while (total_done < bs) and (max_len < self.longest_label):
            x, hn, cn = self._decode_step(x, hn, cn, memory)
            _, x = x.max(1) # x: batch x 1
            x = x.view(-1)
            max_len += 1
            # Store prediction
            for j in range(x.size(0)):
                if not done[j]:
                    token = self.vec2txt([x.data[j]])
                    if token == self.eos:
                        done[j] = True
                        total_done += 1
                    else:
                        preds[j].append(token)
            # Prepare the next input
            x = self.embedding(x).unsqueeze(0) # 1 x batch x hidden

        return preds

    def train(self, x, y):
        self.train_step = True
        if self.use_encoder:
            self.encoder.train()
        self.decoder.train()
        
        x, hn, cn = self._encode(x)
        preds = self._decode_and_train(x, hn, cn, y)
        return preds

    def predict(self, x):
        self.train_step = False
        if self.use_encoder:
            self.encoder.eval()
        self.decoder.eval()
        
        x, hn, cn = self._encode(x)
        preds = self._decode_only(x, hn, cn)
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
        ids = [ex['id'] for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        if len(exs) == 0:
            return (None,)*4

        xs = [ex['text'] for ex in exs]
        x = txt2var(xs)
        y = None
        if 'labels' in exs[0]:
            ys = [' '.join(ex['labels']) for ex in exs]
            y = txt2var(ys, use_offset=False)
        return x, y, ids, valid_inds

    def batch_act(self, observations):
        # observations:
        #       [{'label_candidates': {'office', ...},
        #       'episode_done': False, 'text': 'Daniel ... \nWhere is Mary?',
        #       'labels': ('office',), 'id': 'babi:Task10k:1'}, ...]
        # In joint training, observation examples are randomly selected from all tasks
        batchsize = len(observations)
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        x, y, ids, valid_inds = self.batchify(observations)

        if x is None:
            return batch_reply

        if y is not None:
            preds = self.train(x, y) # [['bedroom', '__NULL__'], ...]
            if self.use_attention and not self.save_atte_done:
                self.save_attention(x.data, y.data, ids)
        else:
            preds = self.predict(x)

        for i in range(len(preds)):
            batch_reply[valid_inds[i]]['text'] = ' '.join(c for c in preds[i]
                            if c != self.eos and c != self.null)

        return batch_reply # [{'text': 'bedroom', 'id': 'Attention'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]

    def save_attention(self, x, y, ids):
        # ids: list of str, self.attn_weight: list of FloatTensor, x and y: LongTensor
        attn_weight = self.attn_weight
        if self.use_cuda:
            attn_weight = [a.cpu() for a in attn_weight]
        attn_weight = [a.numpy() for a in attn_weight]
        attn_weight = np.array(attn_weight).transpose(1,2,0) # batch x source x target
        attn_weights = attn_weight.tolist()
        
        sources = [[self.vec2txt([c]) if c != 0 else '' for c in ex] for ex in x]
        targets = [[self.vec2txt([c]) if c != 0 else '' for c in ex] for ex in y]
        
        for i in range(len(x)):
            id, w, s, t = ids[i], attn_weights[i], sources[i], targets[i]
            self.save_atte_done = self.saver.attention(id, w, s, t)
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
