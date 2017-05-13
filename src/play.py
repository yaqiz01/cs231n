from os import listdir
from os.path import isfile, isdir, join, splitext
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
from path import *
from util import * 
from signdet import *
from speeddet import *
from lightdet import *
import pickle
import time

def roadSignMatching(frame, org, sn):
    sign = cv2.imread(signs[sn])
    sign = cv2.GaussianBlur(sign,(5,5),0)
    img = match(sign, frame, org, draw=True, drawKeyPoint=False, ratioTestPct=0.7, minMatchCnt=5)
    return img

def play(flows, labels, **opts):
    files = [f for f in listdir(opts['path']) if isfile(join(opts['path'], f)) and f.endswith('.png')]
    files = sorted(files)

    if opts['mode'] in ['loadmatch', 'all']:
        matches = mcread(opts['path'])
    if opts['mode'] in ['trainspeed', 'all']:
        headers = loadHeader('{0}/../oxts'.format(opts['path']))

    img = None
    icmp = None
    porg = None
    if (opts['mode'] not in ['trainspeed']):
      plt.figure(dpi=140)
    for i, impath in enumerate(files): 
        fn, ext = splitext(impath)
        if i<opts['startframe']:
            continue
        if opts['endframe']>0 and i>opts['endframe']:
            break
        if opts['numframe']>0 and i>(opts['startframe'] + opts['numframe']):
            break

        root, ext = splitext(impath)
        im = cv2.imread(join(opts['path'], impath), cv2.IMREAD_COLOR)
        org = im.copy()

        opts['fn'] = fn
        if opts['mode'] == 'roadsign':
            im = roadSignMatching(im, org, opts['sign']) 
        elif opts['mode'] == 'loadmatch':
            im,_ = loadMatch(im, org, icmp, fn, matches) 
        elif opts['mode'] == 'detlight':
            im,icmp = detlight(im, org, mode='compare') 
        elif opts['mode'] == 'flow':
            if porg is not None:
                opts['flowmode'] = 'avgflow'
                im = detflow(im, porg, org, **opts)
        elif opts['mode'] == 'trainspeed':
            if porg is not None:
                flow = compFlow(porg, org, **opts)
                flows.append(flow)
                loadLabels(fn, headers, labels, '{0}/../oxts'.format(opts['path']))
        elif opts['mode'] == 'test':
            sp = 30
            sr = 30
            im = cv2.pyrMeanShiftFiltering(im, sp, sr, maxLevel=1)
        elif opts['mode'] == 'all':
            h,w,_ = im.shape
            h = 200
            icmp = np.ones((h,w,3), np.uint8) * 255
            im, (speed, gtspeed, angle, gtangle) = predSpeed(im, porg, org, labels, **opts)
            im, lights = detlight(im, org, mode='label') 
            if opts['detsign']:
                im, signs = loadMatch(im, org, fn, matches) 

            info = []
            info.append('Frame: {0}'.format(fn))
            if speed is None:
                info.append('Predicted speed: X m/s. ground truth: X m/s')
                info.append('Predicted angular velocity: X deg/sec. ground truth: X deg/sec')
                info.append('Current state: X')
            else:
                info.append('Predicted speed: {:02.2f}m/s. ground truth: {:02.2f}m/s'.format(speed,
                    gtspeed))
                info.append('Predicted angular velocity: {:02.4f} deg/sec. ground truth: {:02.4f} deg/sec'.format(angle, gtangle))
                if (speed > 2):
                    if abs(angle)<2:
                        state = 'Forward'
                    elif angle < 0:
                        state = 'Turning Right'
                    else:
                        state = 'Turning Left'
                else:
                    state = 'Still'
                info.append('Current state: {0}'.format(state))
            info.append('Current lights: [{0}]'.format(','.join(lights)))
            if opts['detsign']:
                info.append('Current signs: [{0}]'.format(','.join(signs)))

            h = icmp.shape[0]
            for i, text in enumerate(info):
                coord = (20, h * (i+1)/(len(info)+1))
                fontface = cv2.FONT_HERSHEY_SIMPLEX;
                if iscv2():
                    cv2.putText(img=icmp, text=text, org=coord, fontFace=fontface, fontScale=0.6, 
                        color=bgr('k'), thickness=2, lineType=8);
                elif iscv3():
                    icmp = cv2.putText(img=icmp, text=text, org=coord, fontFace=fontface, fontScale=0.6, 
                        color=bgr('k'), thickness=2, lineType=8);
            loadLabels(fn, headers, labels, '{0}/../oxts'.format(opts['path']))
        porg = org.copy()

        if opts['mode'] in ['trainspeed']:
            continue
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if icmp is not None:
            icmp = cv2.cvtColor(icmp, cv2.COLOR_BGR2RGB)

        if img is None:
            if icmp is not None:
                plt.subplot(2,1,1)
                img = plt.imshow(im)
                plt.subplot(2,1,2)
                imgo = plt.imshow(icmp)
            else:
                img = plt.imshow(im)
        else:
            if icmp is not None:
                imgo.set_data(icmp)
                img.set_data(im)
            else:
                img.set_data(im)
        plt.pause(opts['delay'])
        plt.draw()

def trainModel(opts):
    flows = []
    labels = []
    dirs = [join(KITTI_PATH, d) for d in listdir(KITTI_PATH) if isdir(join(KITTI_PATH, d))]
    for vdir in dirs:
        flows.append([])
        labels.append(dict(vf=[], wu=[]))
        opts['path'] = '{0}/data/'.format(vdir)
        play(flows[-1], labels[-1], **opts)
    return trainSpeed(flows, labels, opts['rseg'], opts['cseg'])

def main():
    usage = "Usage: play [options --path]"
    parser = argparse.ArgumentParser(description='Visualize a sequence of images as video')
    parser.add_argument('--demo', dest='demo', nargs='?', default=1, type=int,
            help='Demo number to run. If --path is set, this option is ignored')
    parser.add_argument('--path', dest='path', action='store', default='',
            help='Specify path for the image files')
    parser.add_argument('--delay', dest='delay', nargs='?', default=0.01, type=float,
            help='Amount of delay between images')
    parser.add_argument('--start-frame', dest='startframe', nargs='?', default=0, type=int,
            help='Starting frame to play')
    parser.add_argument('--end-frame', dest='endframe', nargs='?', default=-1, type=int,
            help='Ending frame to play, -1 for last frame')
    parser.add_argument('--num-frame', dest='numframe', nargs='?', default=-1, type=int,
            help='Number of frame to play, -1 for all frames')
    parser.add_argument('--mode', dest='mode', action='store', default='roadsign')
    parser.add_argument('--rseg', dest='rseg', nargs='?', default=3, type=int,
            help='Number of vertical segmentation in computing averaged flow')
    parser.add_argument('--cseg', dest='cseg', nargs='?', default=11, type=int,
            help='Number of horizontal segmentation in computing averaged flow')
    parser.add_argument('--no-sign', dest='detsign', action='store_false',default=True,
        help='Disable sign detection')
    parser.add_argument('--sign', dest='sign', action='store', default='pedestrian_crossing_left')
    (opts, args) = parser.parse_known_args()

    if (opts.path==''): 
        opts.path = '{0}2011_09_26-{1}/data'.format(KITTI_PATH, opts.demo)

    if (opts.mode=='trainspeed'):
        trainModel(vars(opts))
    else:
        play([], dict(vf=[], wu=[]), **vars(opts))

if __name__ == "__main__":
    main()
