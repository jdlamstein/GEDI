import tensorflow as tf

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import visualization.smooth_gradient_image_placeholder_JDL3 as grad
import numpy as np
import numpy.random as npr
from gedi_config import GEDIconfig


def test_multiply():
    a = 3
    b = 4
    assert a*b == 12

def test_sess():
    a = tf.constant(3)
    b = tf.constant(4)
    c = tf.multiply(a, b)
    with tf.Session() as sess:
        _c = sess.run(c)
        assert _c == 12

def test_list():
    flattened = grad.flatten_list([[1,2],[3,4]])
    sh = np.shape(flattened)
    assert len(sh)==1, 'Shape has more than one dimension'


min_value = 0
max_value = 255
img_dim = np.array([10, 10])

def make_img():
    global min_value, max_value, img_dim

    return npr.random_integers(min_value, max_value, img_dim)


def test_crop_center():
    global img_dim

    img = make_img()
    crop_size = (npr.randint(0, img_dim[0]), npr.randint(0, img_dim[1]))

    cropped = grad.crop_center(img, crop_size)
    sh = np.shape(cropped)

    assert sh == crop_size[::-1]

def test_renormalize():
    global min_value, max_value

    img = make_img()

    renormed = grad.renormalize(img, min_value, max_value)
    flat = grad.flatten_list(renormed) # note test_list() precedes this

    assert max(flat) <= 1
    assert min(flat) >= 0


def test_batcher():
    """Behavior: tests ability of batcher on handling one batch of images, returning numpy, among """

    live_fold = ['/Users/joshlamstein/Documents/GEDI3-master/ScientistLiveDead/BSLive/cache_GEDIhttGlutamate4_C4_9_FITC-DFTrCy5_RFP-DFTrCy5-1.tif']
    dead_fold = ['/Users/joshlamstein/Documents/GEDI3-master/ScientistLiveDead/BSDead/cache_GEDIhttGlutamate4_C4_9_FITC-DFTrCy5_RFP-DFTrCy5-2.tif']

    num_batches = 1 # can try wth multiple but must be compatible with gedi_config or will throw error
    train_min = 0 # unknown usage so far
    train_max = 1 # ditto
    c = GEDIconfig()

    lbls = np.concatenate((np.ones(len(live_fold)), np.zeros(len(dead_fold))))

    batcher = grad.image_batcher(0, num_batches, live_fold + dead_fold, lbls, c, train_max, train_min, 1, True, True) # last arg added: suppress file io

    for i, batch in enumerate(batcher):
        assert np.shape(batch[0])[0] == len(batch[1]) # there is a label for every image
        assert np.shape(batch[0])[1:] == (224, 224, 3) # image dimensions correct/uniform
        assert np.amax(batch[0]) <= train_max
        assert np.amin(batch[0]) >= train_min

    # right num of batches produced?
    # may fail if config batch numbers too high
    assert i == num_batches - 1

