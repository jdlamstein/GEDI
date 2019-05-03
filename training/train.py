#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GEDI. 

To do:
    Explore tensorboard features - embedding, graphs, fit_generator doesn't do histograms. 
    Save tf model
    
    
"""
import param
import os
from models.model import deep
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard

p = param.param()

deep = deep()
model = deep.cnn(input_shape = p.input_shape, output_shape =  p.output_shape)


# Image augmentations - probably won't need them, but just in case. 
# Doesn't include shift in contrast. 
train_datagen = ImageDataGenerator(
        rescale = p.rescale,
        rotation_range = p.rotation_range,
        shear_range = p.shear_range,
        zoom_range = p.zoom_range,
        horizontal_flip = p.horizontal_flip,
        vertical_flip = p.vertical_flip)

test_datagen = ImageDataGenerator(rescale = p.rescale)

train_generator = train_datagen.flow_from_directory(
        p.train_im_dir, 
        target_size = p.target_size,
        batch_size = p.batch_size,
        class_mode = 'binary',
        seed = p.seed)

validation_generator = test_datagen.flow_from_directory(
        p.val_im_dir,
        target_size=p.target_size,
        batch_size = p.batch_size,
        class_mode = 'binary',
        seed = p.seed)

test_generator = test_datagen.flow_from_directory(
        p.test_im_dir,
        target_size = p.target_size,
        batch_size = p.batch_size,
        class_mode = 'binary',
        seed = p.seed
        )

checkpointer = ModelCheckpoint(p.ckpt_file, monitor = 'val_loss', verbose = 0, save_best_only = True)

tb = TensorBoard(log_dir = p.log_dir, 
                 histogram_freq = 0,
                 batch_size = p.batch_size,
                 write_graph = True
        )

callbacks = [checkpointer, tb]

History = model.fit_generator(train_generator, 
                    steps_per_epoch= p.steps_per_epoch, 
                    epochs = p.epochs, 
                    validation_data = validation_generator,
                    validation_steps = p.validation_steps,
                    class_weight = p.class_weight,
                    callbacks = callbacks)

score = model.evaluate_generator(test_generator, steps = p.testing_steps)

predictions = model.predict_generator(test_generator, steps = p.testing_steps)

model.save(os.path.join(p.folder,'saved_models', 'model.h5'))