from _init_paths import *
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
from os.path import isfile, isdir, join, splitext
import argparse
from networks.factory import get_network
import pickle

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

INTERESTED_CLASSES = ['car'] 
CONF_THRESH = 0.8
NMS_THRESH = 0.3

#CLASSES = ('__background__','person','bike','motorbike','car','bus')

sess = None
net = None

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def draw_detections(class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)

def drawObj(ax, scores, boxes, **options):
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        draw_detections(cls, dets, ax, thresh=CONF_THRESH)

def getObj(im, **options):
    path = options['path']
    fn = options['fn']
    bbox_path = '{0}{1}.bbox'.format(SCRATCH_PATH,
      '{0}/{1}'.format(path,fn).replace('/','_').replace('..',''))
    if isfile(bbox_path):
        obj = pickle.load(open(bbox_path, "rb" ))
        scores = obj['scores']
        boxes = obj['boxes']
    else:
        timer = Timer()
        timer.tic()
        initSession(**options)
        scores, boxes = im_detect(sess, net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        obj = {}
        obj['scores'] = scores
        obj['boxes'] = boxes
        pickle.dump(obj , open(bbox_path, "wb"))
    return (scores, boxes)

def objToChannel(im, scores, boxes, **options):
    H,W,_ = im.shape
    
    channel = np.zeros((H,W,len(INTERESTED_CLASSES)))
    for cls_ind, cls in enumerate(CLASSES[1:]):
        if cls not in INTERESTED_CLASSES:
            continue
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            ymin = int(np.ceil(bbox[0]))
            xmin = int(np.ceil(bbox[1]))
            ymax = int(np.floor(bbox[2]))
            xmax = int(np.floor(bbox[3]))
            channel[xmin:xmax, ymin:ymax, INTERESTED_CLASSES.index(cls)] = 1
    return channel

def getObjChannel(im, scores, boxes, **options):
    path = options['path']
    fn = options['fn']
    obj_path = '{0}{1}.obj'.format(SCRATCH_PATH,
      '{0}/{1}'.format(path,fn).replace('/','_').replace('..',''))
    if isfile(obj_path):
        channel = pickle.load(open(obj_path, "rb" ))
    else:
        channel = objToChannel(im, scores, boxes, **options)
        pickle.dump(channel , open(obj_path, "wb"))
    return channel

def demo(image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(args.path, image_name)
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    # timer = Timer()
    # timer.tic()
    # scores, boxes = im_detect(sess, net, im)
    # timer.toc()
    # print ('Detection took {:.3f}s for '
           # '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    options = {'path':args.path, 'fn':image_name}
    scores, boxes = getObj(im, **options)

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

def initSession(**options):
    global sess, net
    if sess is not None and net is not None:
        return
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(options['net'])
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, options['modelpath'])
   
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(options['modelpath'])

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN object detection')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--modelpath', dest='modelpath', help='Model path',
                        default=' ')
    parser.add_argument('--path', dest='path', action='store',
            default='{0}2011_09_26-1/data'.format(KITTI_PATH),
            help='Specify path for the image files')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    global args
    args = parse_args()

    if args.modelpath == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    initSession(**vars(args))
    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                # '001763.jpg', '004545.jpg']

    im_names = [f for f in os.listdir(args.path) if isfile(join(args.path, f)) and f.endswith('.png')]

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}/{}'.format(args.path, im_name)
        demo(im_name)

