# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
demo_path = osp.join(this_dir, '..', 'Faster-RCNN_TF/tools')

# Add caffe to PYTHONPATH
caffe_path = osp.join(demo_path, '..', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(demo_path, '..', 'lib')
add_path(lib_path)


lib_path = osp.join(demo_path,'mftracker')
add_path(lib_path)

DATA_PATH='../driving/'
SIGN_PATH='../signs/'
KITTI_PATH='../kitti/'
SCRATCH_PATH = '../scratch/'
