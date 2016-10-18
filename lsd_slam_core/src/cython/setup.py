#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:39:34 2016

@author: kaihong
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import sys
import numpy
import subprocess

proc_libs = subprocess.check_output("pkg-config --libs opencv".split()).split()
proc_incs = subprocess.check_output("pkg-config --cflags opencv".split()).split()

prefix="/home/kaihong/software/opencv3/"
Libs = "-L"+prefix+"lib -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_highgui -lopencv_imgproc  -lopencv_core "
opencv_libs = [lib for lib in Libs.split()]
opencv_incs = [prefix+"include"]

import eigency

g2o_lib = "g2o_core g2o_stuff csparse cxsparse g2o_solver_csparse g2o_csparse_extension g2o_types_sim3 g2o_types_sba X11".split()
ros_lib = ["roscpp"]
ext_modules = [
    Extension("lsd_slam",
              sources = ["lsd_slam.pyx"],
              language='c++',
              include_dirs = ["../",
                              "../../thirdparty/Sophus/",
                              "/usr/include/eigen3/",
                              numpy.get_include(),]
                              + proc_incs + eigency.get_includes(include_eigen=False),
              library_dirs = [prefix+'lib', "/opt/ros/indigo/lib/"],
              libraries = ["boost_thread"] + g2o_lib + ros_lib,
              extra_objects = ["../../lib/liblsdslam.so"],
              extra_link_args =  proc_libs,
              extra_compile_args = ["-std=c++11"]
)]

setup(
  name = 'lsd_slam for python',
  ext_modules = cythonize(ext_modules),
)
