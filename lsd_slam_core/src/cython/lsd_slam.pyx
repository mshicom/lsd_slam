#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np # for np.ndarray
from libcpp.vector cimport vector
from cpython.ref cimport PyObject
from libcpp cimport bool
from libcpp.string cimport string
# Function to be called at initialization

import cython

np.import_array()

from lsd_slam cimport Frame,DepthMap
from opencv cimport *
from eigency.core cimport *

cdef out_array1d(const float* data_ptr, np.npy_intp* shape):
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32, <void*>data_ptr) if data_ptr != NULL else np.array([])

cdef out_array2d(const float* data_ptr, np.npy_intp* shape):
    return np.PyArray_SimpleNewFromData(2, shape, np.NPY_FLOAT32, <void*>data_ptr) if data_ptr != NULL else np.array([])



cdef class pyFrame(object):
    cdef Frame *thisptr
    cdef np.npy_intp shape[2]
    cdef bool owned
#    def __init__(self, int fid, np.ndarray[np.uint8_t, ndim=2, mode="c"] image, np.ndarray[float, ndim=2, mode="c"] K, double timestamp):
#        self.thisptr = new Frame(fid, image.shape[1], image.shape[0], Map[Matrix3f](K), timestamp, <const unsigned char*>image.data)
#        self.shape[0] = image.shape[0]
#        self.shape[1] = image.shape[1]
#        self.owned = True

    @staticmethod
    cdef create(Frame* ptr):
        p = pyFrame()
        p.thisptr = ptr
        p.shape[0] = ptr.height()
        p.shape[1] = ptr.width()
        p.owned = False
        return p

    def __dealloc__(self):
        if self.owned:
            del self.thisptr

    def image(self, int level=0):
        return out_array2d(self.thisptr.image(level), self.shape)

    def idepth(self, int level=0):
        return out_array2d(self.thisptr.idepth(level), self.shape)

    def idepthVar(self, int level=0):
        return out_array2d(self.thisptr.idepth(level), self.shape)

    def getScaledCamToWorld(self):
        return ndarray(self.thisptr.getScaledCamToWorld().matrix())

    def setTrackingParent(self, pyFrame frame):
        self.thisptr.pose.trackingParent = frame.thisptr.pose

    def setPoseToParent(self, np.ndarray[double, ndim=2, mode="c"] pGr):
        self.thisptr.pose.thisToParent_raw = Sim3(Map[Matrix4d](pGr))

cdef class pyDepthMap(object):
    cdef DepthMap *thisptr
    def __init__(self, int height, int width, np.ndarray[float, ndim=2, mode="c"] K):
        self.thisptr = new DepthMap(width, height, Map[Matrix3f](K))

    def __dealloc__(self):
        del self.thisptr

    def initializeRandomly(self, pyFrame new_frame):
        self.thisptr.initializeRandomly(new_frame.thisptr)

cdef class pySlamSystem(object):
    cdef SlamSystem *thisptr
    cdef bool firstRun
    def __init__(self, int height, int width, np.ndarray[float, ndim=2, mode="c"] K, bool enableSLAM):
        self.thisptr = new SlamSystem(width, height, Map[Matrix3f](K), enableSLAM)
        self.thisptr.setVisualization(NULL) # new ROSOutput3DWrapper(width, height)
        self.firstRun = True
        global displayDepthMap
        global debugDisplay
        global onSceenInfoDisplay
        displayDepthMap = 0
        debugDisplay = 0
        onSceenInfoDisplay = 0

    def __dealloc__(self):
        del self.thisptr

    def finalize(self):
        self.thisptr.finalize()

    def trackFrame(self, np.ndarray[np.uint8_t, ndim=2, mode="c"] image, double timestamp, int idx):
        if self.firstRun:
            self.thisptr.randomInit(<unsigned char*>image.data, timestamp, idx)
            self.firstRun = False
        else:
            self.thisptr.trackFrame(<unsigned char*>image.data, idx, True, timestamp)

    def getCurrentKeyframe(self):
        cdef Frame *kf = self.thisptr.getCurrentKeyframe()
        return pyFrame.create(kf)

    def getCurrentPoseEstimate(self):
        cdef SE3 pos = self.thisptr.getCurrentPoseEstimate()
        return ndarray(pos.matrix())

    def getAllPoses(self):
        cdef vector[pFramePoseStruct, pFramePoseStruct_aligned_allocator] poseStructs = self.thisptr.getAllPoses()
        pos = [ndarray(s.getCamToWorld().matrix()) for s in poseStructs]
        return pos



