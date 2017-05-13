import argparse
import cv2
import six
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

colormap = {} # rgb value from 0 to 1
for name, hex_ in six.iteritems(matplotlib.colors.cnames):
    colormap[name] = matplotlib.colors.hex2color(hex_)

# Add the single letter colors.
for name, hex_ in six.iteritems(matplotlib.colors.ColorConverter.colors):
    colormap[name] = matplotlib.colors.hex2color(hex_)

def rgb(name):
    return [min(c,255) for c in np.floor(np.asarray(colormap[name]) * 256).astype(int)]

def bgr(name):
    [r,g,b] = rgb(name)
    return [b,g,r]

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

# coord is lower left corner of text
def drawLabel(img, label, coord, **options):
    fontface = cv2.FONT_HERSHEY_SIMPLEX;
    scale = 0.6;
    thickness = 2;
    fontcolor = bgr('blue')
    bgdcolor = bgr('w')
    alpha = 0.7
    if 'fontface' in options:
        fontFace = options['fontface']
    if 'scale' in options:
        scale = options['scale']
    if 'thickness' in options:
        thickness = options['thickness']
    if 'fontcolor' in options:
        fontcolor = bgr(options['fontcolor'])
    if 'bgdcolor' in options:
        bdgcolor = options['bdgcolor']

    textSize, baseline= cv2.getTextSize(text=label, fontFace=fontface, fontScale=scale, thickness=thickness);
    blv = coord
    trv = tuple(np.int32(coord)+(np.int32(textSize)+np.array([2,2]))*np.array([1, -1]))
    overlay = img.copy()
    if iscv2():
        thickness = cv2.cv.CV_FILLED
        cv2.rectangle(img=overlay, pt1=blv, pt2=trv, color=bgdcolor, 
            thickness=thickness);
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        blv = tuple(np.array(blv) + np.array([2,-2]))
        cv2.putText(img=img, text=label, org=blv, fontFace=fontface, 
            fontScale=scale, color=fontcolor, thickness=2, lineType=8);
    elif iscv3():
        thickness = cv2.FILLED
        overlay = cv2.rectangle(img=overlay, pt1=blv, pt2=trv, color=bgdcolor, 
            thickness=thickness);
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        blv = tuple(np.array(blv) + np.array([2,-2]))
        img = cv2.putText(img=img, text=label, org=blv, fontFace=fontface, 
            fontScale=scale, color=fontcolor, thickness=2,
            lineType=8);
    return img

def iscv2():
	return cv2.__version__.startswith('2.')
def iscv3():
	return cv2.__version__.startswith('3.')
