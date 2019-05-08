import tensorflow as tf
import visualization.smooth_gradient_image_placeholder_JDL3 as grad
import numpy as np
import numpy.random as npr


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