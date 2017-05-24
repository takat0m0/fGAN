#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import deconv, linear, Layers, batch_norm

class Generator(Layers):
    def __init__(self, name_scopes, in_dim, layer_chanels):
        assert(len(name_scopes) == 2)
        super().__init__(name_scopes)
        self.in_dim = in_dim
        self.layer_chanels = layer_chanels

    def set_model(self, z, batch_size, is_training = True, reuse = False):

        # reshape z
        h = z
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            out_dim = self.in_dim * self.in_dim * self.layer_chanels[0]
            h = linear('_r', h, out_dim)
            h = batch_norm('reshape', h, decay_rate= 0.99,
                           is_training = is_training)
            h = tf.nn.relu(h)
            
        h = tf.reshape(h, [-1, self.in_dim, self.in_dim, self.layer_chanels[0]])

        # deconvolution
        with tf.variable_scope(self.name_scopes[1], reuse = reuse):
            for i, out_chan in enumerate(self.layer_chanels[1:]):
                # get out shape
                h_shape = h.get_shape().as_list()
                out_width = 2 * h_shape[2] 
                out_height = 2 * h_shape[1]
                out_shape = [batch_size, out_height, out_width, out_chan]
                
                # deconvolution
                deconved = deconv(i, h, out_shape, 4, 4, 2)

                # batch normalization
                bn_deconved = batch_norm(i, deconved, 0.99, is_training)

                # activation
                h = tf.nn.relu(bn_deconved)

        return tf.nn.tanh(deconved)

if __name__ == u'__main__':
    g = Generator([u'reshape_z', u'deconvolution'],
                  4, [512, 256, 128, 1])
    z = tf.placeholder(tf.float32, [None, 100])
    c = tf.placeholder(tf.float32, [None, 10])    
    h = g.set_model(c, z, 10)
    h = g.set_model(c, z, 10, True, True)    
    print(h)
