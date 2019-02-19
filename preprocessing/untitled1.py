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
import Match_param as param
import os
import matplotlib.pyplot as plt
import itertools
import random

def rotate(l, n):
    return l[n:] + l[:n]

def rand_shuffle(l, div = 10):
    length = len(l)
    divisions = length // div
    for i in range(div):
        copy = random.shuffle(l[i * divisions : (i+1) * divisions])
        l[i * divisions : (i+1) * divisions] = copy
    return l
        


class record:
    
    def __init__(self, images_dir, masks_dir,  tfrecord_dir):
        self.p = param.param()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.impaths = glob.glob(os.path.join(self.images_dir,'**',  '*.tif') )
        self.maskpaths = glob.glob(os.path.join(self.masks_dir,'**',  '*.tif') )
        self.impaths.sort()
        self.neg_ims = rotate(self.impaths, len(self.impaths)//2)
        self.neg_ims = rand_shuffle(self.neg_ims)
        self.maskpaths.sort()
        self.tfrecord_dir = tfrecord_dir
        self.labels = np.array([np.int32(s.split('_')[-2]) for s in self.impaths])
        self.im_ids = np.array([s.split('_')[-1] for s in self.impaths])
        self.mask_ids = np.array([s.split('_')[-1] for s in self.maskpaths])
        _labels = np.array([np.int32(s.split('_')[-2]) for s in self.maskpaths])
        assert np.all(self.labels==_labels), 'labels don\'t match'
        assert np.all(self.im_ids == self.mask_ids), 'ids don\'t match'
        self.timepoints = np.array([s.split('.')[0] for s in self.im_ids])
        self.timepoints = np.int16(self.timepoints)
        
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
            for i in range(len(self.impaths) - 1):
                print('i', i)
                # one less in range for matching pairs
                if not i % 100:
                    print('Train data:', i) # Python 3 has default end = '\n' which flushes the buffer
#                sys.stdout.flush()
            
                label = self.labels[i]
                label2 = self.labels[i+1]
                timepoint = self.timepoints[i]
                timepoint2 = self.timepoints[i+1]
                imagepath = self.impaths[i]
                if timepoint + 1 == timepoint2 and label == label2:
                    print('timepoint', timepoint, timepoint2)
                    img = self.load_image(self.impaths[i])
                    mask = self.load_image(self.maskpaths[i])
                    img2 = self.load_image(self.impaths[i+1])
                    mask2 = self.load_image(self.maskpaths[i+1])
#                    plt.imshow(mask)
#                    plt.show()
#                    plt.imshow(mask2)
#                    plt.show()
                    feature = {'label': self._int64_feature(label),
                               'timepoint': self._int64_feature(timepoint),
                       'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                       'mask': self._bytes_feature(tf.compat.as_bytes(mask.tostring())),
                       'img2': self._bytes_feature(tf.compat.as_bytes(img2.tostring())),
                       'mask2': self._bytes_feature(tf.compat.as_bytes(mask2.tostring())),
                       'path': self._bytes_feature(tf.compat.as_bytes(imagepath)) }
                    
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            
        sys.stdout.flush()

    def tiff2record_negatives(self, tf_data_name):
        with tf.python_io.TFRecordWriter(os.path.join(self.tfrecord_dir, tf_data_name)) as writer:
            for i in range(len(self.impaths) - 1):
                print('i', i)
                # one less in range for matching pairs
                if not i % 100:
                    print('Train data:', i) # Python 3 has default end = '\n' which flushes the buffer
#                sys.stdout.flush()
            
                label = self.labels[i]
                label2 = self.labels[i+1]
                timepoint = self.timepoints[i]
                timepoint2 = self.timepoints[i+1]
                imagepath = self.impaths[i]
                if timepoint + 1 == timepoint2 and label == label2:
                    print('timepoint', timepoint, timepoint2)
                    img = self.load_image(self.impaths[i])
                    mask = self.load_image(self.maskpaths[i])
                    img2 = self.load_image(self.impaths[i+1])
                    mask2 = self.load_image(self.maskpaths[i+1])
#                    plt.imshow(mask)
#                    plt.show()
#                    plt.imshow(mask2)
#                    plt.show()
                    feature = {'label': self._int64_feature(label),
                               'timepoint': self._int64_feature(timepoint),
                       'image': self._bytes_feature(tf.compat.as_bytes(img.tostring())),
                       'mask': self._bytes_feature(tf.compat.as_bytes(mask.tostring())),
                       'img2': self._bytes_feature(tf.compat.as_bytes(img2.tostring())),
                       'mask2': self._bytes_feature(tf.compat.as_bytes(mask2.tostring())),
                       'path': self._bytes_feature(tf.compat.as_bytes(imagepath)) }
                    
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            
        sys.stdout.flush()
