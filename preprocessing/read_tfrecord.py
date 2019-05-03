#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads tf records and plots images with labels. 
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gedi_param as param
p = param.param()

#data_path = p.X_TRAIN_STR  # path to read tfrecord
data_path = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/cat_dog_train.tfrecord'


feature = {'label': tf.FixedLenFeature([], tf.int64),
           'image': tf.FixedLenFeature([], tf.string)
           }
# Create a list of filenames and pass it to a queue
filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
# Define a reader and read the next record
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# Decode the record read by the reader
features = tf.parse_single_example(serialized_example, features=feature)
# Convert the image data from string back to the numbers
image = tf.decode_raw(features['image'], tf.float32)



# Cast label data into int32
label = tf.cast(features['label'], tf.int32)
# Reshape image data into the original shape
image = tf.reshape(image, [300, 300, 3])

    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
#    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=20, num_threads=1, min_after_dequeue=10, seed = 42)
    
# Initialize all global and local variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:

    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(100):
        lbl, img = sess.run([label, image])
        img = img.astype(np.uint8)
        plt.imshow(img[:,:,0])
        plt.show()
        print('label', lbl)

#            plt.title('cat' if lbl[j]==0 else 'dog')
        
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    