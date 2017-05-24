#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

from tf_util import flatten

def f_star(inputs):
    u'''
    this is f* for Pearson chi squared
    '''
    return 0.25 * tf.square(inputs) + inputs

class Model(object):
    def __init__(self, z_dim, batch_size):

        self.input_size = 256
        self.z_dim = z_dim
        self.batch_size = batch_size
        
        self.lr = 0.0001
        
        # generator config
        gen_layer = [512, 256, 128, 3]
        gen_in_dim = int(self.input_size/2**(len(gen_layer) - 1))

        #discriminato config
        disc_layer = [3, 64, 128, 256]

        # -- generator -----
        self.gen = Generator([u'gen_reshape', u'gen_deconv'],
                             gen_in_dim, gen_layer)

        # -- discriminator --
        self.disc = Discriminator([u'disc_conv', u'disc_fc'], disc_layer)

        
    def set_model(self):

        # -- define place holder -------
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        self.figs= tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3])
        
        # -- generator -----------------
        gen_figs = self.gen.set_model(self.z, self.batch_size, True, False)
        g_logits = self.disc.set_model(gen_figs, True, False)

        self.g_obj =  -tf.reduce_mean(f_star(g_logits))

        self.train_gen  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.g_obj, var_list = self.gen.get_variables())
        
        # -- discriminator --------
        d_logits = self.disc.set_model(self.figs, True, True)
        self.d_obj = -tf.reduce_mean(d_logits) + tf.reduce_mean(f_star(g_logits))
        
        self.train_disc = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.d_obj, var_list = self.disc.get_variables())
        
        # -- for figure generation -------
        self.gen_figs = self.gen.set_model(self.z, self.batch_size, False, True)
        
    def training_gen(self, sess, z_list):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.z: z_list})
        return g_obj
        
    def training_disc(self, sess, z_list, figs):
        _, d_obj = sess.run([self.train_disc, self.d_obj],
                            feed_dict = {self.z: z_list,
                                         self.figs:figs})
        return d_obj
    
    def gen_fig(self, sess, z):
        ret = sess.run(self.gen_figs,
                       feed_dict = {self.z: z})
        return ret

if __name__ == u'__main__':
    model = Model(z_dim = 100, batch_size = 200)
    model.set_model()
    
