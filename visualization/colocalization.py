'''

Compares folders of images for colocalization. Intended for comparing heatmaps of CNN activation layers with original images. Perhaps also the gradients of original
grayscale images.


Refer to:
A practical guide to evaluating colocalization in biological microscopy
Dunn et al.

DATA:
Different Format
\\finkbeinernas01.gladstone.internal\data\robodata\JeremyTEMP\GalaxyTEMP\GEDIhttGlut3b\cache_of_GEDIhttGlutamate3


\\finkbeinernas01.gladstone.internal\data\robodata\JeremyTEMP\GalaxyTEMP\asyntripleA\asyntripleAObjectCropped
'''

import numpy as np
import pandas as pd
import imageio
import glob
import os
from skimage import color

savefile = '/mnt/data/ScientistLiveDead/gradient_images/colocalization.csv'

orig_live_dir = '/mnt/data/ScientistLiveDead/gradient_images/cropped/Live'
orig_dead_dir = '/mnt/data/ScientistLiveDead/gradient_images/cropped/Dead'
grad_live_dir = '/mnt/data/ScientistLiveDead/gradient_images/heatmaps/Live'
grad_dead_dir = '/mnt/data/ScientistLiveDead/gradient_images/heatmaps/Dead'

asyn_dir = '/mnt/finkbeinerlab/data/robodata/JeremyTEMP/GalaxyTEMP/asyntripleA/asyntripleAObjectCropped'

orig_live = glob.glob(os.path.join(orig_live_dir, '*.tif'))
heats_live = glob.glob(os.path.join(grad_live_dir, '*.tif'))

orig_dead = glob.glob(os.path.join(orig_dead_dir, '*.tif'))
heats_dead = glob.glob(os.path.join(grad_dead_dir, '*.tif'))

live_labels = [1 for _ in orig_live]
dead_labels = [0 for _ in orig_dead]

labels = live_labels + dead_labels
orig = orig_dead + orig_live
heats = heats_dead + heats_live


def normalize(im, switchy):
    if switchy == 'grad':
        nim = im / np.max(im)
    elif switchy == 'orig':
        nim = im / 255
    return nim


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
results = []

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
