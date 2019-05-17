from glob import glob
import os

import imageio
from matplotlib import pyplot as plt
import numpy as np
from skimage import color

"""

To do:
 

"""

master_dir = '/Volumes/data/robodata/Gennadi/gradient_images'
o_dir = 'cropped/dead'
g_dir = 'dead_true'
out_file = '/Volumes/data/robodata/Gennadi/gradient_images/dead_true.tif'

thresh = 0.1


def img_to_channel(file, channel):
    img = imageio.imread(file)

    img = img.astype(float)  # cast for operation purposes
    if len(img.shape) > 2: img = color.rgb2gray(img)  # eliminate possible channels
    img -= np.min(img)  # normalize to 0-1
    img /= np.max(img)
    assert img.shape == (224, 224)

    colored = np.zeros((224, 224, 3))
    colored[:, :, channel] = img

    return colored


def make_overlay(orig, grad, thresh):
    assert orig.shape == grad.shape == (224, 224, 3)

    overlay = orig + grad
    for i, row in enumerate(overlay):
        for j, cell in enumerate(row):
            if min(cell[:2]) >= thresh: overlay[i][j] = [1] * 3

    return overlay


def make_triple(file, master_dir, o_dir, g_dir, thresh):
    orig_dir = os.path.join(master_dir, o_dir)
    grad_dir = os.path.join(master_dir, g_dir)

    orig_fn = os.path.join(orig_dir, file)
    grad_fn = os.path.join(grad_dir, file)

    orig_img = img_to_channel(orig_fn, 0)
    grad_img = img_to_channel(grad_fn, 1)
    overlay = make_overlay(orig_img, grad_img, thresh)

    trip = np.hstack((orig_img, grad_img, overlay))
    return trip


def triples_from_dirs(*args):
    assert len(args) == 5
    return [make_triple(file, *args[1:]) for file in args[0]]

def make_table(row_size, *args):
    assert len(args) == 5

    triples = triples_from_dirs(*args)
    if len(triples) % row_size != 0: leftover = row_size - (len(triples) % row_size)
    else: leftover = 0
    triples += [np.ones((224, 224 * 3, 3))] * leftover

    table = []
    for i in range(len(triples) // row_size): table.append(triples[i * row_size:(i + 1) * row_size])
    return table


def combine_cells(table):
    rows = []
    for row in table:
        rows.append(np.hstack(tuple(row)))

    return rows


def combine_all(table):
    rows = combine_cells(table)
    return np.vstack(tuple(rows))


def img_from_fold(num_imgs, row_size, out_file, *args):
    master_dir, o_dir, g_dir, thresh = args

    files = [file.split('/')[-1] for file in glob(os.path.join(master_dir, g_dir, '*.tif'))[:num_imgs]]

    table = make_table(row_size, files, *args)
    img = combine_all(table)

    img *= 255
    img = img.astype(np.uint8)

    imageio.imwrite(out_file, img)


img_from_fold(24, 4, out_file, master_dir, o_dir, g_dir, thresh)