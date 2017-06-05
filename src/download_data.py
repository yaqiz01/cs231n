from _init_paths import *
from os import listdir
from os.path import isfile, isdir, join, splitext, basename, dirname
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse

sample_links = [
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip'
]

full_links = [
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0011/2011_09_26_drive_0011_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0013/2011_09_26_drive_0013_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0014/2011_09_26_drive_0014_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0017/2011_09_26_drive_0017_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0018/2011_09_26_drive_0018_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0048/2011_09_26_drive_0048_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0051/2011_09_26_drive_0051_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0056/2011_09_26_drive_0056_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0057/2011_09_26_drive_0057_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0060/2011_09_26_drive_0060_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0091/2011_09_26_drive_0091_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0106/2011_09_26_drive_0106_sync.zip',
    'http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0113/2011_09_26_drive_0113_sync.zip'
]

def call(cmd):
   print(cmd)
   process = subprocess.call(cmd, shell=True)

def download(**options):
    sample = options['sample']
    path = options['path']
    if sample: links = sample_links 
    else: links = full_links
    for link in links:
        name = link.split('/')[-1].split('.zip')[0]
        data_dir = '{}/{}'.format(path, name)
        if isdir(data_dir):
            print(data_dir + " already exist!")
            continue
        date = name.split('_drive')[0]
        call("wget {} -P {}".format(link, path))
        call("unzip {}/{}.zip -d {}".format(path, name, path))
        call("mv {}/{}/{}/image_02/* {}/{}/{}/ ".format(path, date, name, path, date, name))
        call("rm -r {}/{}/{}/image* {}/{}/{}/velodyne_points".format(path, date, name, path, date, name))
        call("mv {}/{}/{} {}/{}".format(path, date, name, path, name))
        call("rm -r {}/{}".format(path, date))
        call("rm {}/{}.zip".format(path, name))

def main():
    usage = "Usage: plot [options --path]"
    parser = argparse.ArgumentParser(description='Visualize a sequence of images as video')
    parser.add_argument('--path', dest='path', action='store', default=KITTI_PATH,
            help='Specify path for kitti')
    parser.add_argument('--sample', dest='sample', action='store_true',default=False,
        help='Download sample data')
    (options, args) = parser.parse_known_args()

    options = vars(options)

    download(**options)

if __name__ == "__main__":
    main()
