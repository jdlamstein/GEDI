#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:47:49 2018

Parameter file. 

@author: joshlamstein
"""

class param(object):
    def __init__(self):
        
        #data path
        self.raw_folder = '/Volumes/data/robodata/JeremyTEMP/GalaxyTEMP/KS-SOD1-GEDI-1/SortedLiveDead'
        self.folder = '/Volumes/data/robodata/Josh/GEDIDATA'
        self.tfrecord_dir = self.folder + '/tfrecord'
#        
#        self.X_TRAIN = self.folder + '/xtrain.npz'
#        self.Y_TRAIN = self.folder + '/ytrain.npz'
#        self.X_VAL = self.folder + '/xval.npz'
#        self.Y_VAL = self.folder + '/yval.npz'
#        self.X_TEST = self.folder + '/xtest.npz'
#        self.Y_TEST = self.folder + '/ytest.npz'
        
        self.X_TRAIN = 'x_train.tfrecord'
        self.Y_TRAIN = 'y_train.tfrecord'
        self.X_VAL = 'x_val.tfrecord'
        self.Y_VAL = 'y_val.tfrecord'
        self.X_TEST = 'x_test.tfrecord'
        self.Y_TEST = 'y_test.tfrecord'

        
        self.train_im_dir = self.folder + '/train'
        self.test_im_dir = self.folder + '/test'
        self.val_im_dir = self.folder + '/val'
        
        self.ckpt_file = self.folder + '/saved_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        
        #Parameters in model.py
        self.input_shape = (224, 224, 1)
        self.output_shape = 1
        
        #Parameters for ImageDataGenerator
        self.target_size = (224, 224)
        self.rescale = 1./255
        self.shear_range = 0.2
        self.zoom_range = 0.2
        self.rotation_range = 0.2
        self.horizontal_flip = False
        self.vertical_flip = False
        
        # Callbacks
        self.log_dir = self.folder + '/logs'
        
        #Param for flow_from_directory
        self.batch_size = 16
        self.seed = 42
        
        #Param fit_generator
        self.steps_per_epoch = 3 # steps_per_epoch * batch_size = Num images
        self.epochs = 3
        self.validation_steps = 2 # validation_steps * batch_size = num validated_images
        self.class_weight = {0: 1., 1: 1.05}
        
        #Testing
        self.testing_steps = 5