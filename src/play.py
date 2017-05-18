from os import listdir
from os.path import isfile, isdir, join, splitext
import argparse
import numpy as np
import cv2
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from _init_paths import *
from util import *
from signdet import *
from speeddet import *
from objdet import *
from lightdet import *
import pickle
import time
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def roadSignMatching(frame, org, sn):
    sign = cv2.imread(signs[sn])
    sign = cv2.GaussianBlur(sign,(5,5),0)
    img = match(sign, frame, org, draw=True, drawKeyPoint=False, ratioTestPct=0.7, minMatchCnt=5)
    return img

def play(speedXs, labels, **options):
    model = options['model']
    mode = options['mode']
    speedmode = options['speedmode']
    objmask = options['objmask']
    imgchannel = options['imgchannel']

    files = [f for f in listdir(options['path']) if isfile(join(options['path'], f)) and f.endswith('.png')]
    files = sorted(files)

    if mode in ['loadmatch', 'all']:
        matches = mcread(options['path'])
    if mode in ['trainspeed', 'all']:
        headers = loadHeader('{0}/../oxts'.format(options['path']))

    img = None
    icmp = None
    porg = None
    if (mode not in ['trainspeed']):
      plt.figure(dpi=140)
    for i, impath in enumerate(files):
        fn, ext = splitext(impath)
        if i<options['startframe']:
            continue
        if options['endframe']>0 and i>options['endframe']:
            break
        if options['numframe']>0 and i>(options['startframe'] + options['numframe']):
            break

        root, ext = splitext(impath)
        im = cv2.imread(join(options['path'], impath), cv2.IMREAD_COLOR)
        org = im.copy()

        options['fn'] = fn
        if mode == 'roadsign':
            im = roadSignMatching(im, org, options['sign'])
        elif mode == 'loadmatch':
            im,_ = loadMatch(im, org, icmp, fn, matches)
        elif mode == 'detlight':
            im,icmp = detlight(im, org, mode='compare')
        elif mode == 'flow':
            if porg is not None:
                options['flowmode'] = 'avgflow'
                im = detflow(im, porg, org, **options)
        elif mode == 'objdet':
            scores, boxes = getObj(im, **options)
            icmp = getObjChannel(im, scores, boxes, **options)
            icmp = icmp[:,:,0].squeeze() # plot 1 interested channel
        elif mode == 'trainspeed':
            if porg is not None:
                if model=='linear':
                    flow = polarflow(porg, org, **options)
                elif model=='conv':
                    flow = getflow(porg, org, **options)
                speedX = flow
                if objmask:
                    scores, boxes = getObj(im, **options)
                    objchannel = getObjChannel(im, scores, boxes, **options)
                    speedX = np.concatenate((speedX,objchannel), axis=-1)
                if imgchannel:
                    speedX = np.concatenate((speedX,im), axis=-1)
                speedXs.append(speedX)
                print('speedXs.shape={}'.format(np.array(speedXs).shape))
		print('speedmode={} speedX.shape={}'.format(speedmode, np.array(speedX).shape))
                loadLabels(fn, headers, labels, '{0}/../oxts'.format(options['path']))
                return speedXs, labels
        elif mode == 'test':
            sp = 30
            sr = 30
            im = cv2.pyrMeanShiftFiltering(im, sp, sr, maxLevel=1)
        elif mode == 'all':
            h,w,_ = im.shape
            h = 200
            icmp = np.ones((h,w,3), np.uint8) * 255
            im, speed, gtspeed, angle, gtangle = predSpeed(im, porg, org, labels, **options)
            im, lights = detlight(im, org, mode='label')
            if options['detsign']:
                im, signs = loadMatch(im, org, fn, matches)
            scores, boxes = getObj(im, **options)

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
            if options['detsign']:
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
            loadLabels(fn, headers, labels, '{0}/../oxts'.format(options['path']))
        porg = org.copy()

        if mode in ['trainspeed']:
            continue

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if icmp is not None:
            if mode not in ['objdet']:
                icmp = cv2.cvtColor(icmp, cv2.COLOR_BGR2RGB)

        if img is None:
            if icmp is not None:
                imgax = plt.subplot(2,1,1)
                imgoax = plt.subplot(2,1,2)
            else:
                imgax = plt.subplot()

        if icmp is not None:
            if mode in ['objdet']:
                imgo = plt.imshow(icmp, cmap='Greys', interpolation='nearest')
            else:
                imgo = plt.imshow(icmp)
            img = imgax.imshow(im)
        else:
            img = imgax.imshow(im)

        if mode in ['objdet', 'all'] and imgax is not None:
            drawObj(imgax, scores, boxes, **options)

        plt.draw()
        plt.pause(options['delay'])
        if imgax is not None:
            imgax.clear()

def trainModel(**options):
    speedXs = []
    labels = []
    dirs = [join(KITTI_PATH, d) for d in listdir(KITTI_PATH) if isdir(join(KITTI_PATH, d))]
    for vdir in dirs:
        # speedXs.append([])
        # labels.append(dict(vf=[], wu=[]))
        print('before play:  speedXs.shape={}'.format(np.array(speedXs).shape))
        options['path'] = '{0}/data/'.format(vdir)
        speedX, label = play([], dict(vf=[], wu=[]), **options)
        speedXs.append(speedX)
        labels.append(label)
        print('speedXs.shape={}'.format(np.array(speedXs).shape))
    return trainSpeed(speedXs, labels, **options)

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
    parser.add_argument('--mode', dest='mode', action='store', default='roadsign',
            help='Supporting mode: all, loadmatch, roadsign, detlight, flow, test, trainspeed, objdet')
    parser.add_argument('--rseg', dest='rseg', nargs='?', default=3, type=int,
            help='Number of vertical segmentation in computing averaged flow')
    parser.add_argument('--cseg', dest='cseg', nargs='?', default=11, type=int,
            help='Number of horizontal segmentation in computing averaged flow')
    parser.add_argument('--no-sign', dest='detsign', action='store_false',default=True,
        help='Disable sign detection')
    parser.add_argument('--sign', dest='sign', action='store', default='pedestrian_crossing_left')
    parser.add_argument('--model', dest='model', action='store', default='linear',
            help='Specify model for speed detection')
    parser.add_argument('--plot-losses', dest='plot_losses', action='store_true',default=False,
        help='Enable visualization of loss')
    parser.add_argument('--net', dest='net', help='Network to use [vgg16]',
        default='VGGnet_test')
    parser.add_argument('--modelpath', dest='modelpath', help='Model path',
        default='{}/model/VGGnet_fast_rcnn_iter_70000.ckpt'.format(Faster_RCNN_PATH))
    parser.add_argument('--speedmode', dest='speedmode', nargs='?', default=0, type=int,
            help='input mode for speed detection: 0 - flow only, 1 - flow + objmask, 2 - flow + \
            imgchannel, 3 - flow + objmask + imgchannel')
    # parser.add_argument('--gpu', dest='gpu', action='store_true',default=False,
        # help='use gpu to trainspeed')
    (options, args) = parser.parse_known_args()

    if (options.path==''):
        options.path = '{0}2011_09_26-{1}/data'.format(KITTI_PATH, options.demo)

    options = vars(options)

    if (options['speedmode']==0):
        options['objmask'] = False
        options['imgchannel'] = False
    if (options['speedmode']==1):
        options['objmask'] = True
        options['imgchannel'] = False
    if (options['speedmode']==2):
        options['objmask'] = False
        options['imgchannel'] = True
    if (options['speedmode']==3):
        options['objmask'] = True
        options['imgchannel'] = True

    if (options['mode']=='trainspeed'):
        trainModel(**options)
    else:
        play([], dict(vf=[], wu=[]), **options)

if __name__ == "__main__":
    main()
