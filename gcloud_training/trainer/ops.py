#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save operations for hdf5 to ckpt on GCS.
"""

import numpy as np
from tensorflow.python.lib.io import file_io
import os
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import tensorflow.keras.backend as K


def apply_crop(arr, target_size):
    sh = arr.shape
    if sh[1] > target_size[0] and sh[2] > target_size[1]:
        xmid = sh[1]//2
        ymid = sh[2]//2
        
        xmin = xmid - target_size[0]//2
        xmax = xmin + target_size[0]
        ymin = ymid - target_size[1]//2
        ymax = ymin + target_size[1]
        arr = arr[:, xmin:xmax, ymin:ymax, :]
    assert arr.shape[1]==target_size[0], 'X shape doesn\'t match target' + str(arr.shape[1])
    assert arr.shape[2]==target_size[1], 'Y shape doesn\'t match target' + str(arr.shape[2])
    return arr

def apply_reshape(arr, target_shape):
    sh = arr.shape
    if arr.shape[1] != target_shape[0] or arr.shape[2] != target_shape[1]:
        arr = np.reshape(arr, (sh[0],) + target_shape)
    return arr

def apply_rescale(im, scale):
    im = im * scale
    return im

def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
                os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())

def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(
            inputs={'input': model.inputs[0]}, outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tag_constants.SERVING],
                signature_def_map={
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
                })
        builder.save()
        
