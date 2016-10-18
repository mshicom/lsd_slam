#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:13:30 2016

@author: kaihong
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

import os
os.environ["LD_LIBRARY_PATH"] += ";/opt/ros/indigo/lib;"  # for ros to work
from lsd_slam import *

import sys
sys.path.append("/home/kaihong/workspace/gltes")
sys.path.append("/home/nubot/data/workspace/gltes")
sys.path.append("/opt/ros/indigo/lib")
from tools import *

if __name__ == '__main__':
    frames, wGc, K, _ = loaddata1()
    h,w = frames[0].shape

    K = np.ascontiguousarray(K,'f')
    frames = [np.ascontiguousarray(f, np.uint8) for f in frames]
    wGc = [np.ascontiguousarray(relPos(wGc[0], g), np.double) for g in wGc]
#%%
    def test_frame():
        f0 = createPyFrame(0, frames[0], K, 1.2)
        print f0.image()
        print f0.getScaledCamToWorld()

    def test_depthMap():
        f0 = createPyFrame(0, frames[0], K, 1.2)
        dmap = pyDepthMap(h, w, K)
        dmap.initializeRandomly(f0)
        print f0.idepth()

    slam = pySlamSystem(h, w, K)
#    slam.setGradThreshold(10)

    for fid, f in enumerate(frames):
        try:
#            slam.trackFrame(f, fid, fid*0.3)
            slam.importFrame(f, wGc[fid], fid, 0, fid*0.3)
        except:
            pass
        print slam.getCurrentPoseEstimate()
#        plt.waitforbuttonpress()
    print slam.getAllKeyFrames()
