#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save image files as npz. 
"""

import numpy as np
import param
import glob
import os
import imageio

p = param.param()

folders = [p.train_im_dir, p.val_im_dir, p.test_im_dir]
live_dead = ['Live', 'Dead']

saves = [p.X_TRAIN, p.Y_TRAIN, p.X_VAL, p.Y_VAL, p.X_TEST, p.Y_TEST]
skipped = []
for i, f in enumerate(folders):

    for j, g in enumerate(live_dead):

        files = glob.glob(os.path.join(f,g,'*.tif'))
        _hot_array = np.zeros((len(files), 300, 300, 1))
        for idx, h in enumerate(files):
            try:
                _im = imageio.imread(h)
            except:
                print('skipped')
                skipped.append(idx)
            im = np.reshape(_im, (300,300, 1))
            _hot_array[idx, :,:,:] = im
        
        np.savez(saves[i * 2 + j], _hot_array)
        
        