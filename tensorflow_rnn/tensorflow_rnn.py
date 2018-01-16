# ParlAI RNNAgent by TensorFlow
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
import tensorflow as tf
from tensorflow.python.client import device_lib
import copy
import os
import random
import importlib


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
        agent.add_argument('--gpu', type=int, default=-1,
            help='which GPU device to use')

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            # don't enter this loop for shared instantiations
            local_device_protos = device_lib.list_local_devices()
            available = [x.name for x in local_device_protos if x.device_type == 'GPU']
            opt['cuda'] = not opt['no_cuda'] and available
            if opt['cuda']:
                print('[ Using CUDA ]')
                print('[ Using Device:', ', '.join(available), ']')
            else:
                print('[ NO CUDA ]')
            
            self.id = 'RNN'
            self.dict = DictionaryAgent(opt)
            self.observation = {}
            self.rnn_type = opt['rnntype']
            self.hidden_size = opt['hiddensize']
            self.num_layers = opt['numlayers']
            self.dropout_rate = opt['dropout']
            self.learning_rate = opt['learningrate']
            self.path = opt.get('model_file', None)
            self.reuse = None if opt['datatype'] == 'train' else True

            self.create_model()

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            if opt.get('model_file') and os.path.isdir(opt['model_file']):
                print('Loading existing model parameters from ' + opt['model_file'])
                self.load(opt['model_file'])

        self.episode_done = True

    def txt2vec(self, txt):
        return np.array(self.dict.txt2vec(txt)).astype(np.int32)

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

    def create_model(self):
        vs = len(self.dict)
        hs = self.hidden_size   
        nl = self.num_layers
        dr = self.dropout_rate
        lr = self.learning_rate
        reuse = reuse=self.reuse
        self.drop = tf.placeholder(tf.bool)
        
        self.xs = tf.placeholder(tf.int32, [None, None])
        # init = tf.random_uniform_initializer(-1., 1.)
        init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("embeddings", reuse=reuse):
            embeddings = tf.get_variable("var", shape=[vs, hs], initializer=init)
        out = tf.nn.embedding_lookup(embeddings, self.xs) # out: batch x time x hidden
        out = tf.cond(self.drop, lambda: tf.layers.dropout(out, rate=dr), lambda: out)
        if self.rnn_type == 'GRU':
            rnn_cell = tf.nn.rnn_cell.GRUCell(hs, reuse=reuse)
        elif self.rnn_type == 'LSTM':
            rnn_cell = tf.nn.rnn_cell.LSTMCell(hs, reuse=reuse)
        prob = tf.cond(self.drop, lambda: 1.-dr, lambda: 1.)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=prob)
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * nl)
        out, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=out, dtype=tf.float32)
        out = tf.transpose(out, perm=[1, 0, 2]) # out: time x batch x hidden
        out = out[-1] # out: batch x hidden
        logits = tf.layers.dense(out, vs, activation=None, reuse=reuse)
        out = tf.nn.softmax(logits, dim=-1) # out: batch x vocab
        # self.debug = tf.shape(out)

        preds = []
        self.idx = tf.argmax(out, axis=1) # idx: batch
        
        self.ys = tf.placeholder(tf.int32, [None, None])
        ys = tf.transpose(self.ys)[0] # y: batch
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(
                        ys, depth=vs, dtype=tf.float32), logits=logits,)
        loss = tf.reduce_mean(loss)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        #optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_params = optimizer.minimize(loss)
        #gvs = optimizer.compute_gradients(loss)
        #capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
        #self.update_params = optimizer.apply_gradients(capped_gvs)
        

    def train(self, xs, ys):
        """ debug code
        debug = self.sess.run(self.debug,
                        feed_dict={self.xs: xs, self.ys: ys, self.drop: True})
        print("\n\ndebug:\n\n", debug, "\n\n")
        raise SystemExit('Stop for debugging.')
        """
        idx, _ = self.sess.run([self.idx, self.update_params],
                        feed_dict={self.xs: xs, self.ys: ys, self.drop: True})
        preds = [self.vec2txt([i]) for i in idx]
        return preds

    def predict(self, xs):
        idx = self.sess.run(self.idx, feed_dict={self.xs: xs, self.drop: False})
        preds = [self.vec2txt([i]) for i in idx]
        if random.random() < 0.1:
            print('prediction:', preds[0])
        return preds

    def batchify(self, obs):
        """Convert batch observations `text` and `label` to rank 2 tensor `xs` and `ys`
        """
        def txt2np(txt, use_offset=True):
            vec = [self.txt2vec(t) for t in txt]
            max_len = max([len(v) for v in vec])
            arr = np.zeros((len(vec), max_len)).astype(np.int32) # 0 filled rank 2 tensor
            for i, v in enumerate(vec):
                offset = 0
                if use_offset:
                    offset = max_len - len(v) # Right justified
                for j, idx in enumerate(v):
                    arr[i][j + offset] = idx
            return arr # batch x time

        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        if len(exs) == 0:
            return (None,)*3

        xs = [ex['text'] for ex in exs]
        xs = txt2np(xs)
        ys = None
        if 'labels' in exs[0]:
            ys = [' '.join(ex['labels']) for ex in exs]
            ys = txt2np(ys, use_offset=False)
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
        path = self.path if path is None else path
        self.saver.save(self.sess, path + path[path.rfind('/'):])

    def load(self, path):
        self.saver.restore(self.sess, path + path[path.rfind('/'):])
