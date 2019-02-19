#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:26:25 2018

Model copies baseline_vgg16.py. Expect to update it to use more recent model structure. Look at Densenet, Inception, hGru, fGru. 


@author: joshlamstein
"""

import param
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, concatenate, Concatenate, UpSampling2D, Activation
from keras.losses import categorical_crossentropy
from keras import metrics
from keras.initializers import TruncatedNormal
from keras.applications import vgg16, inception_v3

class deep():
    def __init__(self):
        p = param.param()
        self.p = p
    
    def cnn(self, input_shape = (224,224,3), output_shape = 1 ):
        
        bgr_input = Input(input_shape)
        x = Conv2D(64, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv1_1')(bgr_input)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv1_2')(x)
        x = Activation('relu')(x)  
        x = MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same', name = 'pool1')(x)
        
        x = Conv2D(128, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv2_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv2_2')(x)
        x = Activation('relu')(x)  
        x = MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same', name = 'pool2')(x)   
        
        x = Conv2D(256, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv3_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv3_2')(x)
        x = Activation('relu')(x)  
        x = Conv2D(256, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv3_3')(x)
        x = Activation('relu')(x)  
        x = MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same', name = 'pool3')(x)  
    
        x = Conv2D(512, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv4_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv4_2')(x)
        x = Activation('relu')(x)  
        x = Conv2D(512, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv4_3')(x)
        x = Activation('relu')(x)  
        x = MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same', name = 'pool4')(x) 
        
        x = Conv2D(512, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv5_1')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
        x = Conv2D(512, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv5_2')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
        x = Conv2D(512, 3, strides = (1,1), padding = 'same', use_bias= True, name = 'conv5_3')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
        x = MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same', name = 'pool5')(x)
        x = Flatten()(x)
        # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        x = Dense(4096, use_bias = True, kernel_initializer = 'glorot_uniform', bias_initializer = TruncatedNormal(mean = 0.0, stddev = 0.001, seed = 42), name = 'fc6')(x)
        x = Activation('relu')(x)
        x = self.drop_or_batch(x, True, False)
        
        x = Dense(4096, use_bias = True, kernel_initializer = 'glorot_uniform', bias_initializer = TruncatedNormal(mean = 0.0, stddev = 0.001, seed = 42), name = 'fc7')(x)
        x = Activation('relu')(x)
        x = self.drop_or_batch(x, False, True)
        
        x = Dense(4096, use_bias = True, kernel_initializer = 'glorot_uniform', bias_initializer = TruncatedNormal(mean = 0.0, stddev = 0.001, seed = 42), name = 'fc8')(x)
        x = Activation('relu')(x)  
        x = self.drop_or_batch(x)
        
        x = Dense(output_shape, use_bias = True, kernel_initializer = 'glorot_uniform', bias_initializer = TruncatedNormal(mean = 0.0, stddev = 0.001, seed = 42), name = 'fc9')(x)
        output = Activation('softmax', name = 'output')(x)
        model = Model(bgr_input, output)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [metrics.binary_accuracy])
        return model
    
    def vgg(self, input_shape = (224,224,3), output_shape = 1, weights = 'imagenet' ):
        
        base_model = vgg16.VGG16(weights = weights, include_top = False)
        for layer in base_model.layers:
            layer.trainable = False
        base_model.summary()
        
        img = Input(input_shape)
        
        out = base_model(img)
        
        x = Flatten(name = 'flatten')(out)
        x = Dense(4096, activation = 'relu', name = 'fc1')(x)
        x = Dense(4096, activation = 'relu', name = 'fc2')(x)
        prediction = Dense(output_shape, activation = 'softmax', name = 'output')(x)
        
        model = Model(img, prediction)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy' )
        
        return model

        
    def inception_v3(self, input_shape = (299,299,3), output_shape = 1, weights = None):
        base_model = inception_v3.InceptionV3(include_top=False, weights = weights)
        
        for layer in base_model.layers:
            layer.trainable = False
        base_model.summary()
        
        input_img = Input(input_shape)
        
        out = base_model(input_img)
        
        x = GlobalAveragePooling2D()(out)
        prediction = Dense(output_shape, activation = 'softmax', name = 'predictions')(x)
        
        model = Model(input_img, prediction)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        
        return model
        
#        
    
        
    def drop_or_batch(self, x, dropBool = False, batchBool = False):
        if dropBool or batchBool:
            if dropBool:
                x = Dropout(0.5, seed = 42)(x)
            elif batchBool:
                x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
        return x
                
if __name__ == '__main__':
    p = param.param()
    deep = deep()
#    model = deep.cnn(input_shape = p.input_shape, output_shape =  p.output_shape)
    model = deep.vgg(input_shape = p.input_shape, output_shape =  p.output_shape)
#    model = deep.inception_v3(input_shape = p.input_shape, output_shape =  p.output_shape)


    model.summary()
    