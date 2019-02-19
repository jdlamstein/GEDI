n#!/usr/bin/env python3
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
        
#        self.train_im_dir = self.folder + '/train'
#        self.test_im_dir = self.folder + '/test'
#        self.val_im_dir = self.folder + '/val'
#        
#        self.ckpt_file = self.folder + '/saved_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
#        
        #Parameters in model.py
        self.orig_shape = (300,300,1)
        self.input_shape = (300,300,3)
        self.height = self.orig_shape[0]
        self.width = self.orig_shape[1]
        self.channels = self.orig_shape[2]
        self.output_shape = 2
        
        self.num_parallel_calls = 8
        
        #Parameters for ImageDataGenerator
#        self.target_size = (224, 224)
        self.rescale = 1/(2**16 - 1) # for x bit image in processing.dataflow
#        self.shear_range = 0.2
#        self.zoom_range = 0.2
#        self.rotation_range = 0.2
#        self.horizontal_flip = False
#        self.vertical_flip = False
        
#        # Callbacks
#        self.log_dir = self.folder + '/logs'
#        
#        #Param for flow_from_directory
#        self.batch_size = 16
#        self.seed = 42
#        
#        #Param fit_generator
#        self.steps_per_epoch = 3 # steps_per_epoch * batch_size = Num images
#        self.epochs = 3
#        self.validation_steps = 2 # validation_steps * batch_size = num validated_images
#        self.class_weight = {0: 1., 1: 1.}
#        
#        #Testing
#        self.testing_steps = 5
#        
#        self.X_TRAIN_STR = 'gs://rebelbase/GCS/train_Live.tfrecord'
#        self.Y_TRAIN_STR = 'gs://rebelbase/GCS/train_Dead.tfrecord'
#        self.X_VAL_STR = 'gs://rebelbase/GCS/val_Live.tfrecord'
#        self.Y_VAL_STR = 'gs://rebelbase/GCS/val_Dead.tfrecord'
#        self.X_TEST_STR = 'gs://rebelbase/GCS/test_Live.tfrecord'
#        self.Y_TEST_STR = 'gs://rebelbase/GCS/test_Dead.tfrecord'
        
####################GCS########################
        self.TRAIN_STR = 'gs://rebelbase/GCS/train.tfrecord'
        self.VAL_STR = 'gs://rebelbase/GCS/val.tfrecord'
        self.TEST_STR = 'gs://rebelbase/GCS/test.tfrecord'   
        
#####################ANIMALS####################        
#        self.TRAIN_STR = 'gs://rebelbase/animals/cat_dog_train.tfrecord'
#        self.VAL_STR = 'gs://rebelbase/animals/cat_dog_val.tfrecord'
#        self.TEST_STR = 'gs://rebelbase/animals/cat_dog_test.tfrecord'          
#        
#        self.TRAIN_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/train.tfrecord'
#        self.VAL_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/val.tfrecord'
#        self.TEST_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/test.tfrecord'       
        
        
#        self.X_TRAIN_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/train_Live.tfrecord'
#        self.Y_TRAIN_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/train_Dead.tfrecord'
#        self.X_VAL_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/val_Live.tfrecord'
#        self.Y_VAL_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/val_Dead.tfrecord'
#        self.X_TEST_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/test_Live.tfrecord'
#        self.Y_TEST_STR = '/Volumes/data/robodata/Josh/GEDIDATA/tfrecord/test_Dead.tfrecord'        
#        
        self.network_type = 'inception_v3'

        self.EPOCHS = 5
        self.TRAIN_BATCH_SIZE = 16 # 16 # 7436 - 2 2 11 13 13
        self.EVAL_BATCH_SIZE = 16 # 16 # 1860 - 2 2 3 5 31
        self.BUFFER_SIZE = 100 # 1000
        self.TRAIN_STEPS = 595 # 595 # Train 19082 - minus some for corrupted tif / Live 7272 / Dead 11810
#        self.EVAL_STEPS = 10 
        # Val 5917 / Live 2524 / Dead 3393
        # Test length 2952 / Live 1257 / Dead 1965
        self.VALIDATION_STEPS = 182 # 183
        self.EVAL_FREQUENCY = 20
#        self.EVAL_NUM_EPOCHS = 1
#        self.CHECKPOINT_EPOCHS = 10
#        self.TESTING_LENGTH = 5
#        self.MULT_PER_IM = 14
#        self.MULT_SAMPLE = 1
        self.LEARNING_RATE = 0.001
        self.CHECKPOINT_FILE = 'weights.{epoch:02d}.hdf5'
        
        self.MODEL_hdf5 = 'gedi_gcloud.hdf5'