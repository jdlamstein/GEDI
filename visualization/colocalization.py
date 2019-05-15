'''

Compares folders of images for colocalization. Intended for comparing heatmaps of CNN activation layers with original images. Perhaps also the gradients of original
grayscale images.

Get gradient images by running
visualization > smooth_gradient_image_placeholder_JDL3.py
Refer to:
A practical guide to evaluating colocalization in biological microscopy
Dunn et al.

DATA:
Different Format
\\finkbeinernas01.gladstone.internal\data\robodata\JeremyTEMP\GalaxyTEMP\GEDIhttGlut3b\cache_of_GEDIhttGlutamate3


\\finkbeinernas01.gladstone.internal\data\robodata\JeremyTEMP\GalaxyTEMP\asyntripleA\asyntripleAObjectCropped

Here is a link to a data set that was generated using the old cropping script. This means the images are three-paneled (GFP, masked-GFP, and RGEDI left to right) and track the same neuron over time. On the plus side, this means you can follow neurons over time as they die (though the tracking is sometimes poor). But on the downside, this is not the format we use anymore, so you may need to process them differently. Neurons lower on the plate (ie well rows G and H) are treated with higher levels of Glutamate which kill the neurons more quickly.

\\finkbeinernas01.gladstone.internal\data\robodata\JeremyTEMP\GalaxyTEMP\GEDIhttGlut3b\cache_of_GEDIhttGlutamate3


This is a data set with more standard crops (but no time lapse tracking):
\\finkbeinernas01.gladstone.internal\data\robodata\JeremyTEMP\GalaxyTEMP\asyntripleA\asyntripleAObjectCropped

'''

import numpy as np
import pandas as pd
import imageio
import glob
import os
from skimage import color
import matplotlib.pyplot as plt
import stats as st

savefile = '/Users/joshlamstein/Documents/GEDI3-master/ScientistLiveDead/gradient_images/colocalization.csv'

orig_live_dir = '/Users/joshlamstein/Documents/GEDI3-master/ScientistLiveDead/gradient_images/cropped/live'
orig_dead_dir = '/Users/joshlamstein/Documents/GEDI3-master/ScientistLiveDead/gradient_images/cropped/dead'
grad_live_dir = '/Users/joshlamstein/Documents/GEDI3-master/ScientistLiveDead/gradient_images/live_true'
grad_dead_dir = '/Users/joshlamstein/Documents/GEDI3-master/ScientistLiveDead/gradient_images/dead_true'

# asyn_dir = '/mnt/finkbeinerlab/data/robodata/JeremyTEMP/GalaxyTEMP/asyntripleA/asyntripleAObjectCropped' # unused

orig_live = glob.glob(os.path.join(orig_live_dir, '*.tif'))
heats_live = glob.glob(os.path.join(grad_live_dir, '*.tif'))

orig_dead = glob.glob(os.path.join(orig_dead_dir, '*.tif'))
heats_dead = glob.glob(os.path.join(grad_dead_dir, '*.tif'))

live_labels = [1] * len(orig_live)
dead_labels = [0] * len(orig_dead)

# preserve parallel orderings
labels = dead_labels + live_labels
orig = orig_dead + orig_live
heats = heats_dead + heats_live


# unused method
'''def normalize(im, switchy):
    if switchy == 'grad':
        nim = im / np.max(im)
    elif switchy == 'orig':
        nim = im / 255
    return nim'''

def imgPair(orig_file, heat_file):
    im = imageio.imread(orig_file)
    im = color.rgb2gray(im)
    grad = imageio.imread(heat_file)

    im = im / np.max(im)
    grad = grad / np.max(grad)

    return im, grad

def dispImg(img):
    plt.imshow(img)
    plt.show()

# seems to give very low values, possibly denom. calculation off
def pcc(orig, heat):
    '''Pearson correlation coefficient'''
    if np.shape(orig)[-1] == 3:
        orig = color.rgb2gray(orig)
    orig_avg = np.mean(orig)
    heat_avg = np.mean(heat)
    num = np.sum(orig - orig_avg) * np.sum(heat - heat_avg)
    denom = np.sqrt(np.sum((orig - orig_avg) ** 2) * np.sum((heat - heat_avg) ** 2))
    pearson = num / denom
    return pearson


def moc(orig, heat):
    '''Manders overlap coefficient, good for co-occurence, but not intensity correlation.'''
    if np.shape(orig)[-1] == 3:
        orig = color.rgb2gray(orig)
    num = np.sum(orig * heat)
    denom = np.sqrt(np.sum(orig ** 2 * heat ** 2))
    manders = num / denom
    return manders


# def mcc(orig, heat):
#     '''Manders correlation coefficient, fractional overlap, to do.'''
#
"""results = []

for orig_file, heat_file, lbl in zip(orig, heats, labels):
    im = imageio.imread(orig_file)
    grad = imageio.imread(heat_file)
    im = im / np.max(im)

    grad = grad / np.max(grad)
    pearson = pcc(im, grad)
    manders = moc(im, grad)
    results += [[lbl, pearson, manders]]

print(results)

colocalization = pd.DataFrame(results, columns=['label', 'PCC', 'MOC'])

colocalization.to_csv(savefile, index=False)
"""

for orig_file, heat_file, lbl in list(zip(orig, heats, labels))[0:1]:  # only want to test several
    im, grad = imgPair(orig_file, heat_file)
    im_flat, grad_flat = map(st.flatten_np, (im, grad))

    assert np.shape(im) == np.shape(grad)  # same shape so we can use either for iterating counter
    sh = np.shape(im)

    # grab regression line coefficients
    m, b = st.odr_linear(im_flat, grad_flat, plot=False)
    print(m, b)

    print(st.pcc(im_flat, grad_flat))

    lin = lambda x: m * x + b
    pairs_lt = lambda i, g, t: [pair for pair in zip(i, g) if pair[0] < t and pair[1] < lin(t)]
    pairs_gt = lambda i, g, t: [pair for pair in zip(i, g) if pair[0] >= t or pair[1] >= lin(t)]

    t = .01

    _im_flat, _grad_flat = zip(*pairs_lt(im_flat, grad_flat, t))
    print(st.pcc(_im_flat, _grad_flat), len(_im_flat))
    _im_flat, _grad_flat = zip(*pairs_gt(im_flat, grad_flat, t))
    print(st.pcc(_im_flat, _grad_flat), len(_im_flat))

    thresh_img = [[1 if im[i][j] > t else 0 for j in range(sh[1])] for i in range(sh[0])]
    thresh_grad = [[1 if grad[i][j] > lin(t) else 0 for j in range(sh[1])] for i in range(sh[0])]
    thresh_overlay = [[1 if im[i][j] > t and grad[i][j] > lin(t) else 0 for j in range(sh[1])] for i in range(sh[0])]

    plt.imshow(im)
    plt.show()
    plt.imshow(thresh_img)
    plt.show()
    plt.imshow(grad)
    plt.show()
    plt.imshow(thresh_grad)
    plt.show()
    plt.imshow(thresh_overlay)
    plt.show()


    '''blue = lambda i, j: min((im[i][j], grad[i][j])) # what is  smallest intensity @ pixel? i.e. only positive if both channels positive
    overlay = np.array([[[im[i][j], grad[i][j], blue(i, j)] for j in range(sh[1])] for i in range(sh[0])])
    overlay[:, :, 2] /= np.amax(overlay[:, :, 2])

    dispImg(overlay)

    overlay[:, :, :2] = np.zeros(np.shape(overlay[:, :, :2]))

    dispImg(overlay)

    selected = np.array([[[1] * 3 if blue(i, j) > 0.05 else [im[i][j], grad[i][j], 0] for j in range(sh[1])] for i in range(sh[0])])

    dispImg(selected)'''

    # plt.imshow(im)
    # plt.show()
    # plt.imshow(grad)
    # plt.show()
