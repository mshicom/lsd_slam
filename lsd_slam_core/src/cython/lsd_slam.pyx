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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, int fid, np.ndarray[np.uint8_t, ndim=2, mode="c"] image, np.ndarray[float, ndim=2, mode="c"] K, double timestamp):
        height,width = image.shape[:2]

        self.thisptr = new Frame(fid, width, height, FlattenedMap[Matrix, float, _3, _3](K), timestamp, &image[0,0])
        self.shape[0] = image.shape[0]
        self.shape[1] = image.shape[1]

    def __dealloc__(self):
        del self.thisptr

    def image(self, int level=0):
        return out_array2d(self.thisptr.image(level), self.shape)

    def idepth(self, int level=0):
        return out_array2d(self.thisptr.idepth(level), self.shape)

    def idepthVar(self, int level=0):
        return out_array2d(self.thisptr.idepth(level), self.shape)





cdef class pyDepthMap(object):
    cdef DepthMap *thisptr

    def __init__(self, int height, int width, np.ndarray[float, ndim=2, mode="c"] K):
        self.thisptr = new DepthMap(width, height, FlattenedMap[Matrix, float, _3, _3](K))

    def initializeRandomly(self, pyFrame new_frame):
        self.thisptr.initializeRandomly(new_frame.thisptr)

    def __dealloc__(self):
        del self.thisptr


