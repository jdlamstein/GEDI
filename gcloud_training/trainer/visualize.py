#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
At the moment, visualize tf dataset to ensure it's what we expect. 
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import trainer.param as param
import trainer.processing as pro

p = param.param()

dataset = pro.tfdata_generator(filepaths = [p.TRAIN_STR])

iterator = dataset.make_one_shot_iterator()

next_element = iterator.get_next()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:

    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(100):
        img, lbl = sess.run(next_element)
        plt.imshow(img[0, :,:,0])
        plt.show()
        print('label', lbl)