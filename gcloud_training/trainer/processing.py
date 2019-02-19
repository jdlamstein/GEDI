#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:43:13 2019

@author: joshlamstein
"""

import tensorflow as tf
import trainer.param as param
import tensorflow.keras.backend as K





def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                        "label": tf.FixedLenFeature([], tf.int64)}
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    
    # Turn your saved image string into an array
    parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.float32)
    
    parsed_features['image'] = tf.cast(parsed_features['image'], tf.float32)
    
    return parsed_features['image'], parsed_features["label"]

  
def create_dataset(filepaths):
    p = param.param()
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepaths)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function)
    
    # This dataset will go on forever
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle 
#    dataset = dataset.shard(p.num_workers, p.worker_index))
    
#    dataset = dataset.shuffle(p.BUFFER_SIZE)
#    
    # Set the batchsize
    dataset = dataset.batch(p.TRAIN_BATCH_SIZE)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    image = tf.reshape(image, [-1, p.height, p.width, p.channels])
    if p.input_shape[0] < p.height and p.input_shape[1] < p.width:
        image = tf.image.crop_to_bounding_box(image,
            offset_height = p.height//2 - p.input_shape[0]//2,
            offset_width = p.width//2 - p.input_shape[1]//2,
            target_height = p.input_shape[0],
            target_width = p.input_shape[1])
    
    # Create a one hot array for your labels
    label = tf.one_hot(label, p.output_shape)
    
    return image, label

def _tfdata_generator(filepaths):
    def map_func(image, label):
        '''A transformation function'''
        p = param.param()
        x_train = tf.reshape(tf.cast(image, tf.float32), p.orig_shape)
        
        if p.input_shape[0] < p.height and p.input_shape[1] < p.width:
            x_train = tf.image.crop_to_bounding_box(x_train,
                offset_height = p.height//2 - p.input_shape[0]//2,
                offset_width = p.width//2 - p.input_shape[1]//2,
                target_height = p.input_shape[0],
                target_width = p.input_shape[1])
            
        y_train = tf.one_hot(tf.cast(label, tf.uint8), p.output_shape)
        return [x_train, y_train]
    p = param.param()

    dataset = tf.data.TFRecordDataset(filepaths)
    dataset = dataset.map(_parse_function, num_parallel_calls=p.num_parallel_calls)
    dataset  = dataset.map(map_func)
    
    dataset = dataset.repeat()

    
    dataset = dataset.shuffle(p.BUFFER_SIZE)
    
    dataset = dataset.batch(p.TRAIN_BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()

    img, lbl = iterator.get_next()
    return img, lbl
#    while True:
#        next_batch = iterator.get_next()
#        yield K.get_session().run(next_batch)

        
def tfdata_generator(filepaths):
    def _reshape(image, label):
        '''A transformation function'''
        p = param.param()
        # Reshape from string
        x_train = tf.reshape(tf.cast(image, tf.float32), p.orig_shape)
        
#        # Crop
#        if p.input_shape[0] < p.height and p.input_shape[1] < p.width:
#            _x_train = tf.image.crop_to_bounding_box(_x_train,
#                offset_height = p.height//2 - p.input_shape[0]//2,
#                offset_width = p.width//2 - p.input_shape[1]//2,
#                target_height = p.input_shape[0],
#                target_width = p.input_shape[1])
#        
#        # Scale to 0 - 1
#        _x_train = tf.divide(_x_train, 2**16)
#        
#        # Channels
#        if p.input_shape[2] != p.orig_shape[2]:
#            multiply = tf.constant([None, 1,1,p.input_shape[2]])
#            x_train = tf.reshape(tf.tile(_x_train, multiply), [None, p.input_shape[0], p.input_shape[1], p.input_shape[2]] )
#            print(x_train.shape)
#        else:
#            x_train = _x_train
#            print('here')
#        
#        # Set as one hot
#        y_train = tf.one_hot(tf.cast(label, tf.uint8), p.output_shape)
        return [x_train, label]
        
    def _crop(im, label):
        p = param.param()
                # Crop
        if p.input_shape[0] < p.height and p.input_shape[1] < p.width:
            x = tf.image.crop_to_bounding_box(im,
                offset_height = p.height//2 - p.input_shape[0]//2,
                offset_width = p.width//2 - p.input_shape[1]//2,
                target_height = p.input_shape[0],
                target_width = p.input_shape[1])
        else:
            x = im
        return [x, label]
    
    def _scale(image, label):
#        x_train = tf.cast(image, dtype = tf.float32)
#            x_train = tf.multiply(tf.divide(image, 2**16), 2**8)
        if p.network_type == 'vgg16':
            x_train = image[...,::-1]
            print('x_train shape', x_train.shape)
            x_train = x_train / (2.0**16 - 1) * 255
            scaling = tf.constant([103.939,116.779,123.68])
            scaling = tf.broadcast_to(scaling, [x_train.shape[0], x_train.shape[1], 3])
            x_train = x_train - scaling
        elif p.network_type == 'inception_v3':
            x_train = tf.multiply(image, p.rescale) # normalize 16 bit image
        return [x_train, label]

    def _add_channels(image, label):
        p = param.param()

        if p.input_shape[2] != p.orig_shape[2]:
            multiply = tf.constant([1,1, p.input_shape[2]])
            x_train = tf.reshape(tf.tile(image, multiply), (p.input_shape[0], p.input_shape[1], p.input_shape[2]))
            print(x_train.shape)
        else:
            x_train = image
            print('here')   
        return [x_train, label]
    
    def _one_hot(x_train, label):
        p = param.param()
        y_train = tf.one_hot(tf.cast(label, tf.uint8), p.output_shape)
        return [x_train, y_train]
        
    
    p = param.param()

    dataset = tf.data.TFRecordDataset(filepaths)
#    dataset = dataset.map(_parse_function, num_parallel_calls=p.num_parallel_calls)
    dataset = dataset.map(_parse_function)

    dataset  = dataset.map(_reshape)
    dataset  = dataset.map(_crop)
    dataset  = dataset.map(_scale)
    dataset = dataset.map(_add_channels)
    dataset = dataset.map(_one_hot)
    dataset = dataset.shuffle(p.BUFFER_SIZE)

    
    dataset = dataset.repeat()

    
    
    dataset = dataset.batch(p.TRAIN_BATCH_SIZE)

    return dataset     
#def tf_data_generator(images, labels, is_training, batch_size=128):
#    '''Construct a data generator using tf.Dataset'''
#
#    def preprocess_fn(image, label):
#        '''A transformation function to preprocess raw data
#        into trainable input. '''
#        x = tf.reshape(tf.cast(image, tf.float32), (28, 28, 1))
#        y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
#        return x, y
#
#    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#    if is_training:
#        dataset = dataset.shuffle(1000)  # depends on sample size
#
#    # Transform and batch data at the same time
#    dataset = dataset.apply(tf.contrib.data.map_and_batch(
#        preprocess_fn, batch_size,
#        num_parallel_batches=4,  # cpu cores
#        drop_remainder=True if is_training else False))
#    dataset = dataset.repeat()
#    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
#
#    return dataset