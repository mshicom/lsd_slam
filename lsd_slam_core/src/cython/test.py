#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:13:30 2016

@author: kaihong
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from lsd_slam import *



import sys
sys.path.append("/home/kaihong/workspace/gltes")
from tools import *

if __name__ == '__main__':
    frames, wGc, K, _ = loaddata1()
    h,w = frames[0].shape

    K = np.ascontiguousarray(K,'f')
    frames = [np.ascontiguousarray(f, np.uint8) for f in frames]

    dmap = pyDepthMap(h, w, np.ascontiguousarray(K,'f'))
    f0 = pyFrame(0, frames[0], K, 1.2)
    im = f0.image(0)
