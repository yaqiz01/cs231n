#!/usr/bin/env python
import argparse
import numpy as np
import cv2

# Convert video to png

if iscv3(): 
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT 
    CAP_PROP_FPS = cv2.CAP_PROP_FPS
    CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
elif iscv2():
    CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT 
    CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS
    CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

if __name__ == '__main__':
    import sys
    from os.path import *
    usage = "Usage: video2png <videopath> <outputdir>"
    parser = argparse.ArgumentParser(description='Convert video to png')
    parser.add_argument('--start-sec', dest='startsec', type=int, default=0, nargs='?', 
            help='specify path for the image file')
    parser.add_argument('--num-frame', dest='numframe', type=int, default=1, nargs='?', 
            help='specify path for the image file')
    (opts, args) = parser.parse_known_args()
    if (len(args)!=2):
        print(usage)
        exit(-1)
    vi = args[0]
    root, ext = splitext(vi)
    viname = basename(root)        
    outdir = args[1]

    cap = cv2.VideoCapture(vi)

    fcnt = int(cap.get(CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(CAP_PROP_FPS))

    startframe = fps * opts.startsec
    cap.set(CAP_PROP_POS_FRAMES, startframe);

    print('[video2png] Starting at {0}s (frame {1}). Total number of frames: {2}. Saving {3} frames...'.format(
            opts.startsec, startframe, fcnt, opts.numframe))

    readsuc, img = cap.read()
    i = startframe
    while readsuc and i-startframe<opts.numframe:
        outFile = '{0}/{1}_{2}.png'.format(outdir, viname,i)
        cv2.imwrite(outFile,img)
        print(outFile)
        readsuc, img = cap.read()
        i += 1
