#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:19:32 2019

@author: joshlamstein
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert images to TF Records.

Each record for matching has:
    Image
    Mask
    Filepath of image (for tracking it down if there's a problem)
    label - id number of neuron
    timepoint - time point of neuron

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

To do:
    Add img2 and mask2 to pair the data for the siamese network. 
    Save into tf record. 
    Simplest is to save by timepoint, 1->2, 2->3, end -> rand and flag skip.
    Actually simplest to train by family ID. But that isn't the target. 
    Make another tfrecord that programs in negative random samples, verifying that they're not the same family. 

"""

import tensorflow as tf
import imageio
import numpy as np
import sys
import glob
import gedi_param as param
import os
import matplotlib.pyplot as plt
import random


class record:
    
    def __init__(self, images_dir_A, images_dir_B, tfrecord_dir):
        self.p = param.param()
        self.images_dir_A = images_dir_A
        self.images_dir_B = images_dir_B
        self.impaths_A = glob.glob(os.path.join(self.images_dir_A, '*.tif') )
        self.impaths_B = glob.glob(os.path.join(self.images_dir_B, '*.tif') )

        self.tfrecord_dir = tfrecord_dir
        
        live_dead_A = images_dir_A.split('/')[-1]
        if live_dead_A == 'Live':
            label_A = 1
        elif live_dead_A == 'Dead':
            label_A = 0
        else:
            raise ValueError('Last folder A in image directory must be either \'Live\' or \'Dead\'.')

        live_dead_B = images_dir_B.split('/')[-1]
        if live_dead_B == 'Live':
            label_B = 1
        elif live_dead_B == 'Dead':
            label_B = 0
        else:
            raise ValueError('Last folder B in image directory must be either \'Live\' or \'Dead\'.')

        self.labels_A = np.int16(np.ones(len(self.impaths_A)) * label_A)
        self.labels_B = np.int16(np.ones(len(self.impaths_B)) * label_B)
        
        self._impaths = np.array(self.impaths_A + self.impaths_B)
        self._labels = np.append(self.labels_A, self.labels_B)
        assert len(self._impaths) == len(self._labels), 'Length of images and labels do not match.'
        assert len(self.impaths_A) + len(self.impaths_B)==len(self._impaths), 'Summed lengths of image paths do not match'
        self.shuffled_idx = np.arange(len(self._impaths))
        np.random.seed(0)
        np.random.shuffle(self.shuffled_idx)
        print(self.shuffled_idx)

        self.impaths = self._impaths[self.shuffled_idx]
        self.labels = self._labels[self.shuffled_idx]
        
    def load_image(self, im_path):
        img = imageio.imread(im_path)
        #assume it's the correct size, otherwise resize here
        img = img.astype(np.float32)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 
    
    def tiff2record(self, tf_data_name):
        with tf.python_io.TFRecordWriter(os.path.join(self.tfrecord_dir, tf_data_name)) as writer:
            for i in range(len(self.impaths)):
                # one less in range for matching pairs
                if not i % 100:
                    print('Train data:', i) # Python 3 has default end = '\n' which flushes the buffer
#                sys.stdout.flush()
                try:
                    img = self.load_image(self.impaths[i])
    
                    label = self.labels[i]
    
                    feature = {'label': self._int64_feature(label),
                       'image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
                    
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                except:
                    print('skipped')
        print('Saved to ' + os.path.join(self.tfrecord_dir, tf_data_name))
            
        sys.stdout.flush()

if __name__=='__main__':
    p = param.param()
    live_folders = glob.glob('/Volumes/data/robodata/Josh/**/**/Live')
    dead_folders = glob.glob('/Volumes/data/robodata/Josh/**/**/Dead')

    all_folders = live_folders + dead_folders
    print(all_folders)
    
    for live_f, dead_f in zip(live_folders, dead_folders):
        parts_live = live_f.split('/')
        parts_dead = live_f.split('/')
        train_test_val_live = parts_live[-2]
        train_test_val_dead = parts_dead[-2]
        assert train_test_val_live == train_test_val_dead, 'Train test val folder does not match, check folder paths.'
        train_test_val = train_test_val_live
        print(live_f)
        print(dead_f)
        print('Saving to ' + os.path.join(p.tfrecord_dir, train_test_val + '.tfrecord'))
        rec = record(live_f, dead_f, p.tfrecord_dir)
        rec.tiff2record(os.path.join(train_test_val + '.tfrecord' ))

        
    