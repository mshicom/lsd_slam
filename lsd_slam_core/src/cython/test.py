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
sys.path.append("/home/nubot/data/workspace/gltes")
sys.path.append("/opt/ros/indigo/lib")
from tools import *

if __name__ == '__main__':
    frames, wGc, K, _ = loaddata1()
    h,w = frames[0].shape

    K = np.ascontiguousarray(K,'f')
    frames = [np.ascontiguousarray(f, np.uint8) for f in frames]

#    dmap = pyDepthMap(h, w, K)
#    f0 = pyFrame(0, frames[0], K, 1.2)

#    dmap.initializeRandomly(f0)
#    im = f0.image()
#    idep = f0.idepth()
#    print f0.getScaledCamToWorld()


    slam = pySlamSystem(h, w, K, 1)
    for fid, f in enumerate(frames):
        slam.trackFrame(f, fid, fid*0.3)
        print slam.getCurrentPoseEstimate()
        plt.waitforbuttonpress()

