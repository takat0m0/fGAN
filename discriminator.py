#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import conv, linear, Layers, batch_norm, lrelu, flatten

class Discriminator(Layers):
    def __init__(self, name_scopes, layer_channels, output_num = 1):
        assert(len(name_scopes) == 2)
        super().__init__(name_scopes)
        self.layer_channels = layer_channels
        self.output_num = output_num

    def set_model(self, input_img,  is_training = True, reuse = False):
        
        assert(self.layer_channels[0] == input_img.get_shape().as_list()[-1])
        h  = input_img
        
        # convolution
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, out_chan in enumerate(self.layer_channels[1:]):
                h = conv(i, h, out_chan, 4, 4, 2)
                h = lrelu(h)
                
        # fully connect
        h = flatten(h)
        with tf.variable_scope(self.name_scopes[1], reuse = reuse):
            h =  linear('disc_fc', h, self.output_num)
            
        return h

if __name__ == u'__main__':
    dis = Discriminator([u'disc_conv', u'disc_fc'], [1, 64, 128, 256, 512])
                        
    imgs = tf.placeholder(tf.float32, [None, 32, 32, 1])
    h = dis.set_model(imgs)
    h = dis.set_model(imgs, True, True)    
    print(h)
