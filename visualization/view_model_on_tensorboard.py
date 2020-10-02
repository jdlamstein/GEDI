import tensorflow as tf

import os
import time
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from exp_ops.tf_fun import make_dir
from exp_ops.plotting_fun import plot_accuracies, plot_std, plot_cms, plot_pr, \
    plot_cost
from exp_ops.preprocessing_GEDI_images import produce_patch
from gedi_config import GEDIconfig
from models import baseline_vgg16 as vgg16
from tqdm import tqdm


def crop_center(img, crop_size):
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]


def renormalize(img, max_value, min_value):
    return (img - min_value) / (max_value - min_value)


def image_batcher(
        start,
        num_batches,
        images,
        config,
        training_max,
        training_min):
    for b in range(num_batches):
        next_image_batch = images[start:start + config.validation_batch]
        image_stack = []
        for f in next_image_batch:
            # 1. Load image patch
            patch = produce_patch(
                f,
                config.channel,
                config.panel,
                divide_panel=config.divide_panel,
                max_value=config.max_gedi,
                min_value=config.min_gedi).astype(np.float32)
            # print('patch shape 1', np.shape(patch))

            # 2. Repeat to 3 channel (RGB) image
            patch = np.repeat(patch[:, :, None], 3, axis=-1)
            # 3. Renormalize based on the training set intensities
            patch = renormalize(
                patch,
                max_value=training_max,
                min_value=training_min)
            # 4. Crop the center
            # print('patch shape 2', np.shape(patch))

            patch = crop_center(patch, config.model_image_size[:2])
            # print('patch shape 3', np.shape(patch))
            # 5. Clip to [0, 1] just in case
            patch[patch > 1.] = 1.
            patch[patch < 0.] = 0.
            # 6. Add to list
            image_stack += [patch[None, :, :, :]]
        # Add dimensions and concatenate
        start += config.validation_batch
        # print(type(next_image_batch))
        yield np.concatenate(image_stack, axis=0), next_image_batch


def randomization_test(y, yhat, iterations=10000):
    true_score = np.mean(y == yhat)
    perm_scores = np.zeros((iterations))
    lab_len = len(y)
    for it in range(iterations):
        perm_scores[it] = np.mean(
            yhat == np.copy(y)[np.random.permutation(lab_len)])
    p_value = (np.sum(true_score < perm_scores) + 1) / float(iterations + 1)
    return p_value


# Evaluate your trained model on GEDI images
def view_vgg16(
        image_dir,
        model_file,
        output_csv='prediction_file',
        training_max=None):
    print(image_dir)
    #    tf.set_random_seed(0)
    config = GEDIconfig()
    if image_dir is None:
        raise RuntimeError(
            'You need to supply a directory path to the images.')

    combined_files = np.asarray(
        glob(os.path.join(image_dir, '*%s' % config.raw_im_ext)))
    if len(combined_files) == 0:
        raise RuntimeError('Could not find any files. Check your image path.')

    model_file_path = os.path.sep.join(model_file.split(os.path.sep)[:-1])
    print('model file path', model_file_path)
    meta_file_pointer = os.path.join(
        model_file_path,
        'train_maximum_value.npz')
    if not os.path.exists(meta_file_pointer):
        raise RuntimeError(
            'Cannot find the training data meta file: train_maximum_value.npz'
            'Closest I could find from directory %s was %s.'
            'Download this from the link described in the README.md.'
            % (model_file_path, glob(os.path.join(model_file_path, '*.npz'))))
    meta_data = np.load(meta_file_pointer)

    # Prepare image normalization values
    if training_max is None:
        training_max = np.max(meta_data['max_array']).astype(np.float32)
    training_min = np.min(meta_data['min_array']).astype(np.float32)

    # Find model checkpoints
    ds_dt_stamp = re.split('/', model_file)[-2]
    out_dir = os.path.join(config.results, ds_dt_stamp)
    print('out_dir', out_dir)

    # Make output directories if they do not exist
    dir_list = [config.results, out_dir]
    [make_dir(d) for d in dir_list]

    # Prepare data on CPU
    if config.model_image_size[-1] < 3:
        print('*' * 60)
        print(
            'Warning: model is expecting a H/W/1 image. '
            'Do you mean to set the last dimension of '
            'config.model_image_size to 3?')
        print('*' * 60)

    images = tf.placeholder(
        tf.float32,
        shape=[None] + config.model_image_size,
        name='images')

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            vgg = vgg16.model_struct(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.fine_tune_layers)
            vgg.build(
                images,
                output_shape=config.output_shape)
            # print('data dict 1', vgg.data_dict)

        # Setup validation op
        scores = vgg.prob
        preds = tf.argmax(vgg.prob, 1)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())
    test_writer = tf.summary.FileWriter('/mnt/data/GEDI_RESULTS/logs')

    # Loop through each checkpoint then test the entire validation set
    ckpts = [model_file]
    ckpt_yhat, ckpt_y, ckpt_scores, ckpt_file_array = [], [], [], []
    print('-' * 60)
    print('Beginning evaluation')
    print('-' * 60)

    if config.validation_batch > len(combined_files):
        print('Trimming validation_batch size to %s (same as # of files).' % len(combined_files))
        config.validation_batch = len(combined_files)

    for idx, c in tqdm(enumerate(ckpts), desc='Running checkpoints'):
        dec_scores, yhat, file_array = [], [], []
        # Initialize the graph

        #        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(
                tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()))

            # Set up exemplar threading
            new_saver = tf.train.import_meta_graph(os.path.join(c + '.meta'))
            new_saver.restore(sess, c)
            all_vars = tf.trainable_variables()
            print(all_vars)
            for v in all_vars:
                if v.name=='cnn/fc8/fc8_biases:0':
                # if v.name=='cnn/conv5_3/conv5_3_filters:0':
                    v_ = sess.run(v)
                    print(v_)
                    print('Printed {}'.format(v.name))
            if 0:
                saver.restore(sess, c)
                start_time = time.time()
                num_batches = np.floor(
                    len(combined_files) / float(
                        config.validation_batch)).astype(int)
                for image_batch, file_batch in tqdm(
                        image_batcher(
                            start=0,
                            num_batches=num_batches,
                            images=combined_files,
                            config=config,
                            training_max=training_max,
                            training_min=training_min),
                        total=num_batches):
                    feed_dict = {
                        images: image_batch
                    }
                    sc, tyh = sess.run(
                        [scores, preds],
                        feed_dict=feed_dict)
                    dec_scores = np.append(dec_scores, sc)
                    yhat = np.append(yhat, tyh)
                    file_array = np.append(file_array, file_batch)
                ckpt_yhat.append(yhat)
                ckpt_scores.append(dec_scores)
                ckpt_file_array.append(file_array)
                print('Batch %d took %.1f seconds' % (
                    idx, time.time() - start_time))
                test_writer.add_graph(sess.graph)
                # print('data dict', vgg.data_dict)


def print_weights(ckpt):
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        sess.run(
            tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()))
        new_saver = tf.train.import_meta_graph(os.path.join(ckpt + '.meta'))
        # print(tf.train.latest_checkpoint('./'))
        new_saver.restore(sess, ckpt)
        all_vars = tf.trainable_variables()
        print(all_vars)
        for v in all_vars:
            v_ = sess.run(v)
            print(v_)


def view_meta_file(metafile):
    tf.train.import_meta_graph(metafile)
    for n in tf.get_default_graph().as_graph_def().node:
        print(n)
    with tf.Session() as sess:
        test_writer = tf.summary.FileWriter('/mnt/data/GEDI_RESULTS/logs2')
        test_writer.add_graph(sess.graph)


#    sess.close()

if __name__ == '__main__':
    image_dir = '/mnt/finkbeinerlab/robodata/JaslinTemp/GalaxyData/LINCS-diMNs/LINCS072017RGEDI-A/Galaxy/CroppedImages-Voronoi/F12'
    model_file = '/home/jlamstein/Documents/pretrained_weights/trained_gedi_model/model_58600.ckpt-58600'
    metafile = '/home/jlamstein/Documents/pretrained_weights/trained_gedi_model/model_58600.ckpt-58600.meta'
    ckpt = '/home/jlamstein/Documents/pretrained_weights/trained_gedi_model'
    view_vgg16(image_dir, model_file, training_max=None)
    # view_meta_file(metafile)
    # print_weights(model_file)