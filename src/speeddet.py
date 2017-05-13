import argparse
from os import listdir
from os.path import isfile, join, splitext
from ntpath import basename
import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *
from path import *
import csv
from multiprocessing.pool import ThreadPool
from functools import partial
from sklearn import datasets, linear_model
import pickle
        
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, bgr('g'))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 2, bgr('g'), -1)
    return img

def draw_avgflow(img, avgflow):
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
    gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow, avgflow = getflow(prevgray, gray, **options)
    if flowmode == 'allflow':
        frame = draw_flow(frame, flow)
    elif flowmode == 'avgflow':
        frame = draw_flow(frame, flow)
        frame = draw_avgflow(frame, avgflow)
    return frame

def getflow(prevgray, gray, **options):
    rseg = options['rseg']
    cseg = options['cseg']
    path = options['path']
    fn = options['fn']

    # gray = np.round(np.random.rand(4,4,2)*3)
    h, w = gray.shape[:2]
    rstride = h / rseg
    cstride = w / cseg

    flow_path = '{0}{1}.flow'.format(SCRATCH_PATH, 
      '{0}/{1}'.format(path,fn).replace('/','_').replace('..',''))
    
    if 'flowMap' in options and flow_path in options['flowMap']:
      flow = options['flowMap'][flow_path] 
    elif isfile(flow_path):
      flow = pickle.load(open(flow_path, "rb" )) 
      if 'flowMap' in options:
        options['flowMap'][flow_path] = flow
    else:
      if iscv2():
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
      elif iscv3():
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
      pickle.dump(flow , open(flow_path, "wb"))

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
    return flow, avgflow

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
    
def compFlow(prev, cur, **options):
    gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow, avgflow = getflow(prevgray, gray, **options)
    
    cplx = avgflow[:,:,0] + avgflow[:,:,1] * 1j
    cplx = cplx.flatten()
    mag = np.absolute(cplx)
    ang = np.angle(cplx)
    return mag.tolist() + ang.tolist()

def predSpeed(im, prev, cur, labels, **options):
    if prev is None:
        return im, (None, None, None, None)
    parampath = SCRATCH_PATH
    if 'parampath' in options:
        parampath = options['parampath']

    with open('{0}/parameters.txt'.format(parampath), 'r') as paramfile:
        rseg, cseg = paramfile.readline().split(',')
        rseg = int(rseg)
        cseg = int(cseg)
        coef_speed = paramfile.readline().split(',')
        coef_speed = np.array(map(float, coef_speed))
        coef_angle = paramfile.readline().split(',')
        coef_angle = np.array(map(float, coef_angle))
    options['rseg'] = rseg
    options['cseg'] = cseg
    flow = compFlow(prev, cur, **options)
    regr_speed = linear_model.LinearRegression()
    regr_speed.coef_ = coef_speed
    regr_speed.intercept_ = True 
    speed = regr_speed.predict([flow])[0]
    gtspeed = labels['vf'][-1]
    regr_angle = linear_model.LinearRegression()
    regr_angle.coef_ = coef_angle
    regr_angle.intercept_ = True 
    angle = regr_angle.predict([flow])[0]
    gtangle = np.rad2deg(labels['wu'][-1])

    return im, (speed, gtspeed, angle, gtangle) 

def trainSpeed(flows, labels, rseg, cseg, **options):
    pctTrain = 0.8
    parampath = SCRATCH_PATH
    if 'parampath' in options:
        parampath = options['parampath']

    numTest = int(round(len(flows)*(1-pctTrain)))
    # Split the data into training/testing sets
    X_train = []
    X_test = []
    vly_train = []
    vly_test = []
    agy_train = []
    agy_test = []
    for fl,lb in zip(flows, labels):
        mk = np.random.randint(10, size=len(fl)) < pctTrain*10
        # X_train += fl[:-numTest]
        # X_test += fl[-numTest:]
        fl = np.array(fl)
        X_train += fl[mk].tolist()
        X_test += fl[~mk].tolist()
        lb['vf'] = np.array(np.array(lb['vf']))
        vly_train += lb['vf'][mk].tolist()
        vly_test += lb['vf'][~mk].tolist()
        lb['wu'] = np.array(np.array(lb['wu']))
        agy_train += lb['wu'][mk].tolist()
        agy_test += lb['wu'][~mk].tolist()
    
    # Create linear regression object
    regr_speed = linear_model.LinearRegression(fit_intercept=True)
    # Train the model using the training sets
    regr_speed.fit(X_train, vly_train)
    vlmse = np.mean((regr_speed.predict(X_test) - vly_test) ** 2)
    vlvar = regr_speed.score(X_test, vly_test)

    agy_train = np.rad2deg(agy_train)
    agy_test = np.rad2deg(agy_test)
    # Create linear regression object
    regr_angle = linear_model.LinearRegression(fit_intercept=True)
    # Train the model using the training sets
    regr_angle.fit(X_train, agy_train)
    agmse = np.mean((regr_angle.predict(X_test) - agy_test) ** 2)
    agvar = regr_speed.score(X_test, agy_test)
    
    # write coefficients into a file
    with open('{0}/parameters.txt'.format(parampath), 'w') as paramfile:
        paramfile.write(','.join(map(str, [rseg, cseg])) + '\n')
        paramfile.write(','.join(map(str, regr_speed.coef_)) + '\n')
        paramfile.write(','.join(map(str, regr_angle.coef_)) + '\n')

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Speed mean squared error: {:.2f}, Speed variance score: {:.2f}, Angle mean squared error:{:.2e}, Angle variance score: {:.2f}".format(vlmse, vlvar, agmse, agvar))
    return (vlmse, vlvar, agmse, agvar)
    
# def main():

# if __name__ == "__main__":
    # main()
