#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model copies baseline_vgg16.py. Expect to update it to use more recent model structure. Look at Densenet, Inception, hGru, fGru. 
To do:
    Model isn't learning anything
    Check normalization
    Initialized weights
    More data
    learning rate
    depth of model
    Activation function
    bad gradients
@author: joshlamstein
"""

import trainer.param as param
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, concatenate, Concatenate, UpSampling2D, Activation
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import metrics
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.applications import vgg16, inception_v3

class deep:
    def __init__(self,network_type, input_shape = None, output_shape = None):
        p = param.param()
        self.p = p
        if network_type =='cnn':
            self.model = self.cnn(input_shape, output_shape)
        elif network_type =='vgg16':
            self.model = self.vgg(input_shape, output_shape)
        elif network_type =='inception_v3':
            self.model = self.inception_v3(input_shape, output_shape)
        else:
            raise ValueError('Invalid network type string')
    
    def cnn(self, input_shape = (224,224,1), output_shape = None ):
#    def cnn(self, tensor= None, output_shape = 2, label = None):
#        print(tensor.dtype)
#        print(tensor.shape)
#        inputs = Input(tensor = tensor)
#        print(inputs.dtype)
#        print(inputs.shape)
        inputs = Input(input_shape)
        
        x = Conv2D(32, 3, activation = 'relu',strides = (1,1), padding = 'same', use_bias= True, name = 'conv1_1')(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same', name = 'pool1')(x)
        
        x = Conv2D(64, 3,activation = 'relu', strides = (1,1), padding = 'same', use_bias= True, name = 'conv2_1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same', name = 'pool2')(x)   
        
        x = Conv2D(128, 3, activation = 'relu',strides = (1,1), padding = 'same', use_bias= True, name = 'conv3_1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same', name = 'pool3')(x)  
    
        x = Conv2D(64, 3, activation = 'relu',strides = (1,1), padding = 'same', use_bias= True, name = 'conv4_1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same', name = 'pool4')(x) 
        
        x = Conv2D(32, 3, activation = 'relu',strides = (1,1), padding = 'same', use_bias= True, name = 'conv5_1')(x)
#        x = BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001)(x)
        x = MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same', name = 'pool5')(x)
        x = Flatten()(x)
        # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        x = Dense(1024, use_bias = True, activation = 'relu', kernel_initializer = 'glorot_uniform', bias_initializer = TruncatedNormal(mean = 1.0, stddev = 0.001, seed = 42), name = 'fc6')(x)
#        x = self.drop_or_batch(x, True, False)
        
        
        output = Dense(output_shape, use_bias = True, activation = 'softmax', kernel_initializer = 'glorot_uniform', bias_initializer = TruncatedNormal(mean = 1.0, stddev = 0.001, seed = 42), name = 'output')(x)
        model = Model(inputs, output)
#        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [metrics.binary_accuracy], target_tensors = [label])
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [metrics.categorical_accuracy])

        model.summary()
        return model
    
    def vgg(self, input_shape = (224,224,3), output_shape = 2, weights = 'imagenet' ):
        '''
        Trained with BGR color format, values -128 to 128. 
        '''
        base_model = vgg16.VGG16(weights = weights, include_top = False, input_shape = input_shape)
        for layer in base_model.layers[:5]:
            layer.trainable = False
        base_model.summary()
        
        img = Input(input_shape)
        
        out = base_model(img)
        
        x = Flatten(name = 'flatten')(out)
        x = Dense(4096, activation = 'relu', name = 'fc1')(x)
        x = Dense(4096, activation = 'relu', name = 'fc2')(x)
        prediction = Dense(output_shape, activation = 'softmax', name = 'output')(x)
        
        model = Model(img, prediction)
        model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'] )
        
        return model

        
    def inception_v3(self, input_shape = (299,299,3), output_shape = 1, weights = 'imagenet'):
        base_model = inception_v3.InceptionV3(include_top=False, weights = weights)
        
        for layer in base_model.layers:
            layer.trainable = False
        base_model.summary()
        
        input_img = Input(input_shape)
        
        out = base_model(input_img)
        
        x = GlobalAveragePooling2D()(out)
        x = Dense(4096, activation = 'relu', name = 'fc1')(x)
        x = Dense(512, activation = 'relu', name = 'fc2')(x)
        prediction = Dense(output_shape, activation = 'softmax', name = 'predictions')(x)
        
        model = Model(input_img, prediction)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        return model
        
#        
    
        
    def drop_or_batch(self, x, dropBool = False, batchBool = False):
        if dropBool or batchBool:
            if dropBool:
                x = Dropout(0.8, seed = 42)(x)
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
    