import argparse
from os import listdir
from os.path import isfile, join, splitext, dirname, basename
from ntpath import basename
import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *
from path import *
from objdet import *
import csv
from multiprocessing.pool import ThreadPool
from functools import partial
from sklearn import datasets, linear_model
import pickle
from linearmodel import *

def lookup(speedmode):
    if (speedmode==0):
        includeflow = True
        includeobj = False
        includeimg = False
    elif (speedmode ==1):
        includeflow = True
        includeobj = True
        includeimg = False
    elif (speedmode==2):
        includeflow = True
        includeobj = False
        includeimg = True
    elif (speedmode==3):
        includeflow = True
        includeobj = True
        includeimg = True
    elif (speedmode==4):
        includeflow = False
        includeobj = False
        includeimg = True
    return includeflow, includeobj, includeimg

def drawflow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, bgr('g'))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 2, bgr('g'), -1)
    return img

def drawAvgflow(img, avgflow):
    h, w = img.shape[:2]
    hs, ws = avgflow.shape[:2]
    hstep = h/hs
    wstep = w/ws
    # print(h,w, hstep, wstep, hs, ws, h/hstep, w/wstep)
    y, x = np.mgrid[hstep/2:hstep*hs:hstep, wstep/2:wstep*ws:wstep].reshape(2,-1).astype(int)
    ys, xs = np.mgrid[0:hs, 0:ws].reshape(2,-1).astype(int)
    fx, fy = avgflow[ys,xs].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, bgr('r'), thickness=2)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 4, bgr('r'), -1)
    return img

def detflow(frame, prev, cur, **options):
    flowmode = options['flowmode']
    flow = getflow(prev, cur, **options)
    avgflow = getAvgflow(flow, **options)
    if flowmode == 'allflow':
        frame = drawflow(frame, flow)
    elif flowmode == 'avgflow':
        frame = drawflow(frame, flow)
        frame = drawAvgflow(frame, avgflow)
    return frame

def getflow(prev, cur, **options):
    path = options['path']
    fn = options['fn']
    flow_path = '{0}{1}.flow'.format(SCRATCH_PATH,
      '{0}/{1}'.format(path,fn).replace('/','_').replace('..',''))

    if 'flowMap' in options and flow_path in options['flowMap']:
        flow = options['flowMap'][flow_path]
    elif isfile(flow_path):
        if options['checkcache']: return
        # print('load {}'.format(flow_path))
        flow = pickle.load(open(flow_path, "rb" ))
        if 'flowMap' in options:
            options['flowMap'][flow_path] = flow
    else:
        # print('recompute {}'.format(flow_path))
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        if iscv2():
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif iscv3():
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        pickle.dump(flow , open(flow_path, "wb"))
    return flow

def getAvgflow(flow, **options):
    rseg = options['rseg']
    cseg = options['cseg']
    h, w = flow.shape[:2]
    rstride = h / rseg
    cstride = w / cseg
    # flow = gray
    avgflow = np.ndarray((rseg, cseg, 2), dtype=flow.dtype)
    for ir in range(0, rseg):
        rstart = ir*rstride
        rend = min(rstart+rstride, h)
        for ic in range(0, cseg):
            cstart = ic*cstride
            cend = min(cstart+cstride, w)
            grid = flow[rstart:rend, cstart:cend]
            avgflow[ir, ic] = np.mean(grid, axis=(0,1))
    return avgflow

def loadHeader(path):
    headers = {}
    with open('{0}/dataformat.txt'.format(path), 'r') as dataformat:
        for i, line in enumerate(dataformat):
            headers[line.split(':')[0]] = i
    return headers

def loadLabels(fn, headers, labels, labelpath):
    with open('{0}/data/{1}.txt'.format(labelpath, fn), 'r') as data:
        line = data.readline()
        vals = line.split(' ')
        for key in labels:
            labels[key].append(float(vals[headers[key]]))

def loadData(framePaths, **options):
    H,W,C = options['inputshape']
    model = options['model']
    speedXs = []
    path = dirname(framePaths[0])
    headers = loadHeader('{0}/../oxts'.format(path))
    labels = dict(vf=[], wu=[])
    for framePath in framePaths:
        path = dirname(framePath) + "/"
        fn, ext = splitext(basename(framePath))
        options['path'] = path
        options['fn'] = fn
        speedX = np.zeros((H,W,0))
        speedmode = options['speedmode']
        includeflow, includeobj, includeimg = lookup(speedmode)
        options['checkcache'] = False
        if includeflow:
            if model=='linear':
                flow = polarflow(None, None, **options)
            elif model=='conv':
                flow = getflow(None, None, **options)
            speedX = np.concatenate((speedX,flow), axis=-1)
        if includeobj:
            objchannel = getObjChannel(None, **options)
            speedX = np.concatenate((speedX,objchannel), axis=-1)
        if includeimg:
            im = cv2.imread(framePath, cv2.IMREAD_COLOR)
            speedX = np.concatenate((speedX,im), axis=-1)
        if speedX.shape != (H,W,C):
            raise Exception('data input shape={} not equals to expected shape!{}'.format(
                (H,W,C), speedX.shape))
        speedXs.append(speedX)
        # print('speedmode={} speedX.shape={}'.format(speedmode, np.array(speedX).shape))
        loadLabels(fn, headers, labels, '{0}/../oxts'.format(path))
    speedXs = np.reshape(np.array(speedXs), (-1, H,W,C))
    speedYs = np.reshape(np.array(labels['vf']), (-1, 1))
    return ([speedXs, speedYs])

def polarflow(prev, cur, **options):
    flow = getflow(prev, cur, **options)
    avgflow = getAvgflow(flow, **options)

    # cplx = avgflow[:,:,0] + avgflow[:,:,1] * 1j
    # cplx = cplx.flatten()
    # mag = np.absolute(cplx)
    # ang = np.angle(cplx)
    # return mag.tolist() + ang.tolist()

    cplx = avgflow[:,:,0] + avgflow[:,:,1] * 1j
    H,W = cplx.shape
    mag = np.reshape(np.absolute(cplx), (H,W,1))
    ang = np.reshape(np.angle(cplx), (H,W,1))
    return np.concatenate((mag,ang), axis=-1)

def predSpeed(im, prev, cur, labels, **options):
    model = options['model']
    speedmode = options['speedmode']
    if prev is None:
        return im, None, None, None, None
    flow = polarflow(prev, cur, **options)
    includeflow, includeobj, includeimg = lookup(speedmode)
    if includeobj:
        pass #TODO
    else:
        X_test = flow

    if model=='linear':
        speed, angle = linearRegressionModelTest(X_test, **options)

    gtspeed = labels['vf'][-1]
    gtangle = np.rad2deg(labels['wu'][-1])
    return im, speed, gtspeed, angle, gtangle

def trainSpeed(frameFns, **options):
    from convmodel import ConvModel
    pcttrain = options['pcttrain']
    model = options['model']
    print('Start training speed ...')

    frameFns = np.array(frameFns)
    N = frameFns.shape[0]
    numTrain = int(round(N*pcttrain))
    mask = np.zeros(N, dtype=bool)
    mask[:numTrain] = True
    np.random.shuffle(mask)
    frameTrain = frameFns[mask]
    frameVal = frameFns[~mask]
    print("frameTrain.shape={} framVal.shape={}".format(
        frameTrain.shape, frameVal.shape))
    if model=='linear':
        pass
    elif model=='conv':
        conv_model = ConvModel(options)
        conv_model.train(frameTrain, frameVal)
        conv_model.close()

def trainSpeedOld(speedXs, labels, **options):
    """
    Train a linear regression model for speed detection

    :param speedXs: averaged dense flow of multiple videos. speedXs[video, frame, flowmag+flowang]
    :param labels: a dictionary of true labels of each frame
    :returns: this is a description of what is returned
    :raises keyError: raises an exception
    """
    pcttrain = 0.8
    model = options['model']
    print('Start training speed ...')

    numTrain = int(round(len(speedXs)*(pctTrain)))
    # Split the data into training/testing sets
    X_train = []
    X_test = []
    vly_train = []
    vly_test = []
    agy_train = []
    agy_test = []
    for speedX,lb in zip(speedXs, labels):
        mask = np.zeros(len(speedX), dtype=bool)
        mask[:numTrain] = True
        np.random.shuffle(mask)
        # mask = np.random.randint(10, size=len(speedX)) < pctTrain*10

        speedX = np.array(speedX)
        xtrain = speedX[mask]
        xtest = speedX[~mask]
        X_train += xtrain.tolist()
        X_test += xtest.tolist()
        lb['vf'] = np.array(lb['vf'])
        vly_train += lb['vf'][mask].tolist()
        vly_test += lb['vf'][~mask].tolist()
        lb['wu'] = np.array(lb['wu'])
        agy_train += lb['wu'][mask].tolist()
        agy_test += lb['wu'][~mask].tolist()

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print("X_train.shape={} X_test.shape={}".format(X_train.shape, X_test.shape))
    vly_train = np.array(vly_train)
    vly_test = np.array(vly_test)
    agy_train = np.array(agy_train)
    agy_test = np.array(agy_test)

    if model=='linear':
        vlmse, vlvar, agmse, agvar = linearRegressionModelTrain(
                X_train, X_test,
                vly_train, vly_test,
                agy_train, agy_test,
                **options)
    elif model=='conv':
        conv_model = ConvModel(options)
        vlmse, vlvar, agmse, agvar = conv_model.train(X_train, X_test,
                vly_train,vly_test, agy_train, agy_test)
        conv_model.close()
        # clear old variables
        # tf.reset_default_graph()
    # The mean squared error
    print("Speed mean squared error: {:.2f}, Speed variance score: {:.2f}, Angle mean squared error:{:.2e}, Angle variance score: {:.2f}".format(vlmse, vlvar, agmse, agvar))
    return (vlmse, vlvar, agmse, agvar)

# def main():

# if __name__ == "__main__":
    # main()
