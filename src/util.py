from __future__ import division

import sys
import time
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

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]
