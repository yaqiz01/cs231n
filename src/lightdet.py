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
import matplotlib.patches as patches

labels = dict(r='red_light', y='yellow_light', g='green_light')
fontcolors = dict(r='red', y='chocolate', g='green')

def findLight(lc, cmks, img, **options):
    # mode='compare'
    mode='label'
    if 'mode' in options:
        mode = options['mode']

    Tsq = 0.15
    msk = cmks[lc]
    if iscv2():
        contours, hierarchy = cv2.findContours(msk,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    elif iscv3():
        _, contours, hierarchy = cv2.findContours(msk,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    lightbounds = []
    for cnt in contours:
        (_, (rw, rh), _) = cv2.minAreaRect(cnt.copy())
        if (abs(rw-rh) > rw*Tsq):
            continue

        cvxhull = cv2.convexHull(cnt.copy())
        # hull = np.squeeze(cvxhull)
        # if (len(hull.shape)==1): # convex hull is a point
            # continue
        # maxpxl = np.amax(hull, 0) 
        # minpxl = np.amin(hull, 0)
        # h, w = maxpxl - minpxl 

        x,y,w,h = cv2.boundingRect(cnt)
        minpxl = (x,y)
        maxpxl = (x+w,y+h)

        minX, minY = minpxl
        maxX, maxY = maxpxl
        bounds = [minX, maxX, minY, maxY]

        mg = 0.4
        cond = True
        if (lc=='r'):
            surbounds = [minX, maxX, int(minY+(1+1*mg)*h), int(maxY+(1+1*mg)*h)]
            scolors = surroundColors(msk, bounds, surbounds, cmks, img)
            cond = cond and ('k' in scolors or 'y' in scolors)
            # print('v-1', scolors, bounds, surbounds)
            surbounds = [minX, maxX, int(minY+(2+2*mg)*h), int(maxY+(2+2*mg)*h)]
            scolors = surroundColors(msk, bounds, surbounds, cmks, img)
            cond = cond and 'k' in scolors
            # print('v-2', scolors, bounds, surbounds)
            # surbounds = [int(minX-(2+1*mg)*w), int(maxX-(2+1*mg)*h), minY, maxY]
            # scolors = surroundColors(msk, bounds, surbounds, cmks, img)
            # cond = cond and 'k' not in scolors
            # # print('h-1', scolors, bounds, surbounds)
            # surbounds = [int(minX+(1+1*mg)*w), int(maxX+(1+1*mg)*w), minY, maxY]
            # scolors = surroundColors(msk, bounds, surbounds, cmks, img)
            # cond = cond and 'k' not in scolors
            # print('h+1', scolors, bounds, surbounds)
        elif (lc=='y'):
            surbounds = [minX, maxX, int(minY-(1+1*mg)*h), int(maxY-(1+1*mg)*h)]
            scolors = surroundColors(msk, bounds, surbounds, cmks, img)
            cond = cond and 'k' in scolors or 'r' in scolors
            # print('v+1', scolors, bounds, surbounds)
            surbounds = [minX, maxX, int(minY+(1+1*mg)*h), int(maxY+(1+1*mg)*h)]
            scolors = surroundColors(msk, bounds, surbounds, cmks, img)
            cond = cond and 'k' in scolors
            # print('v-1', scolors, bounds, surbounds)
        elif (lc=='g'):
            surbounds = [minX, maxX, int(minY-(2+2*mg)*h), int(maxY-(1+1*mg)*h)]
            scolors = surroundColors(msk, bounds, surbounds, cmks, img)
            cond = cond and 'k' in scolors
        if cond:
            if mode=='compare':
                ()
                # img = cv2.drawContours(img, [cvxhull], contourIdx=-1, color=rgb('g'), thickness=2)
            elif mode=='label':
                if iscv2():
                    cv2.drawContours(img, [cvxhull], contourIdx=-1, color=rgb('g'),
                        thickness=1)
                elif iscv3():
                    img = cv2.drawContours(img, [cvxhull], contourIdx=-1, color=rgb('g'),
                        thickness=1)
            coord = (maxX + 4, (minY + maxY)/2)
            img = drawLabel(img, labels[lc], coord, fontcolor=fontcolors[lc])  

            lightbounds.append(bounds)
    return img, lightbounds

def surroundColors(curmsk, bounds, surbounds, cmks, img):
    region = getPatch(curmsk, bounds)
    colors = [] 
    for cn in cmks:
        msk = cmks[cn]
        # msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, region)
        surround = getPatch(msk, surbounds)

        # if cn=='y':
            # print(surround.sum(), surround.size)
            # print(colors)
            # plt.subplot(1,3,1)
            # region = getPatch(img, bounds)
            # plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            # #plt.imshow(setPatch(curmsk, bounds, 1), cmap=plt.cm.binary)
            # plt.subplot(1,3,2)
            # plt.imshow(surround, cmap=plt.cm.binary)
            # plt.subplot(1,3,3)
            # surround = getPatch(img, surbounds)
            # plt.imshow(cv2.cvtColor(surround, cv2.COLOR_BGR2RGB))
            # plt.show()

        if (surround.sum() >= 0.4 * surround.size):
            colors.append(cn)

    # if (curmsk==cmks['y']).all():
        # plt.subplot(2,1,1)
        # region = getPatch(img, bounds)
        # plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        # #plt.imshow(setPatch(curmsk, bounds, 1), cmap=plt.cm.binary)
        # plt.subplot(2,1,2)
        # surround = getPatch(img, surbounds)
        # plt.imshow(cv2.cvtColor(surround, cv2.COLOR_BGR2RGB))
        # plt.show()

    return colors

def getPatch(img, bounds):
    [minX, maxX, minY, maxY] = bounds
    h,w = img.shape[0:2]
    return img[max(minY, 0):min(maxY, h), max(minX, 0):min(maxX, w)] 

def setPatch(img, bounds, val):
    [minX, maxX, minY, maxY] = bounds
    h,w = img.shape[0:2]
    cp = img.copy()
    cp[max(minY, 0):min(maxY, h), max(minX, 0):min(maxX, w)] = val
    return cp

def detlight(img, org, **options):
    # mode='compare'
    mode='label'
    if 'mode' in options:
        mode = options['mode']

    frame= cv2.GaussianBlur(org,(5,5),0)
    h,w,_ = org.shape

    xr = np.int32(frame[:,:,2])
    xg = np.int32(frame[:,:,1])
    xb = np.int32(frame[:,:,0])

    Tr = 130
    Ty = 130
    Tg = 130
    Tk = 40

    setR = ((xr - xg) >= Tr) & ((xr - xb) >= Tr)
    setY = ((xg - xb) >= 90) & ((xr - xb) >= 130)
    setG = ((xg - xr) >= Tg) & ((xb - xr) >= Tg)
    setK = (xr < Tk) & (xg < Tk) & (xb < Tk)
    
    xhr = np.zeros(org.shape, dtype=np.uint8)
    xhr[:,:,2] = 255 #r
    xhy = np.zeros(org.shape, dtype=np.uint8)
    xhy[:,:,2] = 255 #r
    xhy[:,:,1] = 255 #g
    xhg = np.zeros(org.shape, dtype=np.uint8)
    xhg[:,:,1] = 255 #g
    xhg[:,:,0] = 255 #b
    xhw = np.ones(org.shape, dtype=np.uint8) * 255
    xhk = np.zeros(org.shape, dtype=np.uint8)

    mkr = np.ones((h,w), dtype=np.uint8) * setR 
    mky = np.ones((h,w), dtype=np.uint8) * setY 
    mkg = np.ones((h,w), dtype=np.uint8) * setG 
    mkk = np.ones((h,w), dtype=np.uint8) * setK 

    cmks = dict(r=mkr, y=mky, g=mkg, k=mkk)

    if (mode=='compare'):
        for i in range(3):
            img[:,:,i] =(
                           setR * xhr[:,:,i] 
                         + setG * xhg[:,:,i] 
                         + setY * xhy[:,:,i]
                         + setK * xhk[:,:,i]
                         + (~setR & ~setG & ~setY & ~setK) * xhw[:,:,i]
                         )

    img, rbounds = findLight('r', cmks, img, **options)
    img, ybounds = findLight('y', cmks, img, **options)
    img, gbounds = findLight('g', cmks, img, **options)

    # circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=h/8,
                                # param1=100,param2=20,minRadius=0,maxRadius=150)
    
    # circles = np.uint16(np.around(circles))
    # for i in circles[0,:]:
        # # draw the outer circle
        # cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
        # # draw the center of the circle
        # cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

    if mode=='compare':
        return img, org
    elif mode=='label':
        # icmp = options['icmp']
        lights = []
        if len(rbounds)!=0:
            lights.append('Red')
        if len(ybounds)!=0:
            lights.append('Yellow')
        if len(gbounds)!=0:
            lights.append('Green')
        # h = icmp.shape[0]
        # coord = (20, h*2/4)
        # fontface = cv2.FONT_HERSHEY_SIMPLEX;
        # icmp = cv2.putText(img=icmp, text=text, org=coord, fontFace=fontface, 
            # fontScale=0.6, color=bgr('k'), thickness=2, lineType=8);
        return img, lights 

def main():
    usage = "Usage: match [options --mode]"
    parser = argparse.ArgumentParser(
        description='detect traffic light in a frame or a video')
    parser.add_argument('--start-frame', dest='startframe', nargs='?', default=0, type=int,
            help='Starting frame to play')
    parser.add_argument('--end-frame', dest='endframe', nargs='?', default=-1, type=int,
            help='Ending frame to play, -1 for last frame')
    parser.add_argument('--num-frame', dest='numframe', nargs='?', default=-1, type=int,
            help='Number of frame to play, -1 for all frames')
    parser.add_argument('--mode', dest='mode', action='store', default='detectone')
    parser.add_argument('--numthread', dest='numthread', nargs='?', default=8, type=int,
            help='Number of thread to match roadsigns')
    parser.add_argument('--path', dest='path', action='store',
            default='{0}/2011_09_26_1/data/'.format(KITTI_PATH))
    parser.add_argument('--file', dest='file', action='store',
            default='{0}/2011_09_26-3/data/0000000004.png'.format(KITTI_PATH))
    (opts, args) = parser.parse_known_args()

    if (opts.mode == 'detectall'):
        print('TODO') 
    elif (opts.mode == 'detectone'):
        img = cv2.imread(opts.file)
        img,org = detlight(img, img.copy(), mode='compare') 
        org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(dpi=140)
        plt.subplot(2,1,1)
        plt.imshow(org)
        plt.subplot(2,1,2)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()

if __name__ == "__main__":
    main()
