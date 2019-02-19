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
import cv2


class record:
    
    def __init__(self, images_dir, tfrecord_dir):
        self.p = param.param()

        self.tfrecord_dir = tfrecord_dir
        
        self._impaths = np.array(glob.glob(os.path.join(images_dir, '*.jpg')))
        self._labels = np.array([0 if i.split('/')[-1].split('.')[0]=='cat'  else 1 for i in self._impaths])
        
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
    
    def tiff2record(self, tf_data_name, begin, finish):
        with tf.python_io.TFRecordWriter(os.path.join(self.tfrecord_dir, tf_data_name)) as writer:
            for i in range(begin, finish):
                # one less in range for matching pairs
                if not i % 100:
                    print('Train data:', i) # Python 3 has default end = '\n' which flushes the buffer
#                sys.stdout.flush()
                try:
                    img = self.load_image(self.impaths[i])
                    img = cv2.resize(img, (300,300) )
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
    images_dir = '/Users/joshlamstein/Desktop/dogs_vs_cats/train'

    rec = record(images_dir, p.tfrecord_dir)
    print('Saving to ' + os.path.join(p.tfrecord_dir,'cat_dog_train.tfrecord'))
    rec.tiff2record(os.path.join(p.tfrecord_dir,'cat_dog_train.tfrecord'), 0, 5000)
    print('Saving to ' + os.path.join(p.tfrecord_dir,'cat_dog_val.tfrecord'))
    rec.tiff2record(os.path.join(p.tfrecord_dir,'cat_dog_val.tfrecord'), 5000,7500)  
    print('Saving to ' + os.path.join(p.tfrecord_dir,'cat_dog_test.tfrecord'))
    rec.tiff2record(os.path.join(p.tfrecord_dir,'cat_dog_test.tfrecord'), 7500, 10000)  
         


        
    