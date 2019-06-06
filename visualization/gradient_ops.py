"""
Make a function that takes two image inputs: original image and gradient image
    - they may be different sizes, they need to be the same size. I think we crop the original image
    -Should be output of crop_center
Concatenate two images into single numpy array (224,224) + (224,224) -> (448, 224)
Save to folder and plot
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
"""import gedi_config as config

import numpy.random as npr
def make_img(dim, maxcolor=255):
    return npr.random_integers(0, maxcolor, dim)"""


def rgb_to_gray(img):
    """
    
    Args:
        img: normalizes image to grayscale for compatibility

    Returns: 224x224 image tensor

    """
    
    assert np.shape(img) == (224, 224, 3)

    # weighted multiplication to grayscale
    return np.average(img, weights=[0.299, 0.587, 0.114], axis=2)

# can consider changing to use gediconfig normalization?
def normalize_imgs(original, gradient):
    """
    
    Args:
        original: 2d numpy (already rgb->gray), of some floating-point ([0,x<1]) normalization
        gradient: 2d numpy of some integer ([0, x~=255]) normalization

    Returns: tuple of: original image normalized with max color of gradient image, gradient image

    """
    
    assert len(np.shape(original)) == 2,     'Original image does not have exactly 2 dimensions'
    assert min(np.shape(original)) >= 224,   'Original image too small'
    assert np.shape(gradient) == (224, 224), 'Gradient image not 224x224'
    
    # normalize bottom/min color to 0
    original -= np.amin(original)
    gradient -= np.amin(gradient)
    
    # normalize top color to  that of gradient image
    original /= np.amax(original)
    original *= np.amax(gradient)
    
    return (original, gradient)

def crop_center(img, crop_size):
    """
    
    Args:
        img:       2d numpy arr of dimensions (y>=crop_size[0], x>=crop_size[1])
        crop_size: dims to cut array down to (center slice taken)

    Returns:

    """
    
    x, y = img.shape[:2]
    cx, cy = crop_size
    # beginning of slices found
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    
    return img[starty:starty + cy, startx:startx + cx] # sliced size of cy, cx as needed

def concat_imgs(original, gradient):
    """
    
    Args:
        original: 2d original numpy image; gets cropped to fit with gradient
        gradient: 2d gradient image

    Returns: outputs combined graph of images w/ dimension 448, 224

    """
    original = crop_center(original, [224, 224])
    return np.hstack((original, gradient))

def plot_save_imgs(original, gradient, fName=None, pred=None, actual=None):
    """

    Args:
        original: original numpy image (3 color channels)
        gradient: computed visualization gradient (grayscale, 224x224)
        fName: filename for storing combined img, grad plot
        pred: model prediction of neuron state
        actual: actual (label) state

    Returns:

    """

    # f = plt.figure()

    plt.imshow(concat_imgs(*normalize_imgs(rgb_to_gray(original), gradient)))

    status = lambda n: 'live' if n == 1. else 'dead'
    if ((pred, actual).count(None) == 0):
        plt.title("Predicted: {0}; Actual: {1}".format(*map(status, (pred, actual))))

    if fName != None:
        plt.savefig(fName)
    
    plt.show()
    plt.close(f)

