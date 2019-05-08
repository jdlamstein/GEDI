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

import numpy.random as npr
def make_img(dim, maxcolor=255):
    return npr.random_integers(0, maxcolor, dim)

def crop_center(img, crop_size):
    """Center crop images."""
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]

def concat_imgs(original, gradient):
    """Stacks images together"""
    original = crop_center(original, [224, 224])
    return np.hstack((original, gradient))

def plot_save_imgs(original, gradient, fName=None):
    # f = plt.figure() # used in smooth_gradient..?
    plt.imshow(concat_imgs(original, gradient))
    if fName != None:
        plt.savefig(fName)
    plt.show()


dummy_imgs = (make_img([300, 300]), make_img([224, 224], maxcolor=100), './test.png')
#plot_save_imgs(*dummy_imgs)