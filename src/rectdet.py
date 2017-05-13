import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
from path import *

# Detect rectangle in image

def main(opts):

    global img
    img = cv2.imread(opts.filepath, cv2.IMREAD_COLOR)

    # contour detection
    # img = cv2.blur(img,(3,3));
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray,127,255,0)
    # image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # lcontours = []
    # for c in contours:
        # peri = cv2.arcLength(c, True)
        # if peri>50 and peri < 200:
            # lcontours.append(c)
        # approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # plt.ion()
    # for i, c in enumerate(lcontours):
        # img = cv2.drawContours(img, lcontours, contourIdx=i, color=(0,255,0), thickness=1)
        # plt.figure()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # _ = raw_input("Press any key to continue")
        # plt.show()
        # plt.close()
    # return

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=gray,threshold1=50,threshold2=200,apertureSize=3)
    plt.imshow(edges, cmap=plt.cm.binary)
    plt.show()
    return

    minLineLength = 30
    maxLineGap = 5
    global lines
    lines = cv2.HoughLinesP(edges,rho=10,theta=np.pi/180*10,threshold=80,
                minLineLength=minLineLength,maxLineGap=maxLineGap)
    global line
    for line in lines:
        x1,y1,x2,y2 = line[0]
        img = cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def detcircle(opts):
    cimg = cv2.imread(opts.filepath, cv2.IMREAD_COLOR)
    cimg = cv2.medianBlur(cimg,5)
    img = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=80,param2=30,minRadius=3,maxRadius=50)
    
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    return cimg

if __name__ == "__main__":
    usage = "Usage: detrect [options --file]"
    parser = argparse.ArgumentParser(description='Detect rectangles in an image')
    parser.add_argument('--file', dest='filepath', action='store', 
            default='../kitti/2011_09_26_1/data/0000000000.png',
            help='specify path for the image file')

    (opts, args) = parser.parse_known_args()

    # main(opts)
    fig = plt.figure(figsize=(14, 10))
    plt.subplot(2,1,1)
    opts.filepath = '/Users/Yaqi/ee368/kitti/2011_09_26-1/data/0000000075.png'
    cimg = detcircle(opts)
    plt.imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
    plt.title('(a)')
    plt.axis('off')
    plt.subplot(2,1,2)
    opts.filepath = '/Users/Yaqi/ee368/kitti/2011_09_26-2/data/0000000010.png'
    cimg = detcircle(opts)
    plt.imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB))
    plt.title('(b)')
    plt.axis('off')
    plt.tight_layout(pad=0.1, h_pad=0.3)
    plt.savefig('{0}/circlesign1.png'.format(SCRATCH_PATH), dpi=fig.dpi)
    plt.show()
