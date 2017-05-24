#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np

from model import Model
from util import get_figs, dump_figs

if __name__ == u'__main__':

    # data_dir
    data_dir = u'train_input'

    # parameter

    epoch_num = 100

    z_dim = 100
    batch_size = 200
    
    # make model
    print('-- make model --')
    model = Model(z_dim, batch_size)
    model.set_model()

    # get_data
    print('-- get figs--')
    figs = get_figs(data_dir)
    print('num figs = {}'.format(len(figs)))

    # training
    print('-- begin training --')
    num_one_epoch = len(figs) //batch_size

    nrr = np.random.RandomState()
    def shuffle(x):
        rand_ix = nrr.permutation(x.shape[0])
        return x[rand_ix]
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epoch_num):
            epoch_figs = shuffle(figs)
            print('** epoch {} begin **'.format(epoch))
            g_obj = 0.0
            d_obj = 0.0
            for step in range(num_one_epoch):
                
                # get batch data
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                batch_figs = epoch_figs[step * batch_size: (step + 1) * batch_size]
                # train
                d_obj += model.training_disc(sess, batch_z, batch_figs)
                g_obj += model.training_gen(sess, batch_z)
                
                if step%10 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
                    tmp_figs = model.gen_fig(sess, batch_z)
                    dump_figs(np.asarray(tmp_figs), 'sample_result')
                    
            print('epoch:{}, d_obj = {}, g_obj = {}'.format(epoch,
                                                            d_obj/num_one_epoch,
                                                            g_obj/num_one_epoch))
            saver.save(sess, './model.dump')
