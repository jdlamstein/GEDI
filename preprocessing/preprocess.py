#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copies images from .../live + .../dead to target_folder/train, .../test, .../val. 

Uses:
    p.raw_folder
    p.folder
"""

import glob
import os
import shutil
import param
import numpy as np

p = param.param()

for x in glob.glob(os.path.join(p.raw_folder,'Live', '*.tif')):
    r = np.random.rand()
    
    if r < 0.1:
        dst = os.path.join(p.folder, 'test', 'Live')
    elif r < 0.3:
        dst = os.path.join(p.folder, 'val', 'Live')
    else:
        dst = os.path.join(p.folder, 'train', 'Live')
    src = x
    if not os.path.exists(dst):
        os.makedirs(dst)         
    shutil.copy2(src, dst)
    
for x in glob.glob(os.path.join(p.raw_folder, 'Dead', '*.tif')):
    r = np.random.rand()
    
    if r < 0.1:
        dst = os.path.join(p.folder, 'test', 'Dead')
    elif r < 0.3:
        dst = os.path.join(p.folder, 'val', 'Dead')
    else:
        dst = os.path.join(p.folder, 'train', 'Dead')
    src = x
    if not os.path.exists(dst):
        os.makedirs(dst)         
    shutil.copy2(src, dst)