import argparse
from matplotlib import pyplot as plt
import matplotlib
import importlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from play import *

def foo(opts):
    i = opts['i']
    j = opts['j']
    return (i*j, i+j)

def exp(opts):
    opts['mode'] = 'trainspeed'
    rsegs = range(1,15,1) 
    csegs = range(1,24,1) 
    nr = len(rsegs)
    nc = len(csegs)
    rsegs, csegs = np.meshgrid(rsegs, csegs, sparse=False, indexing='ij')
    rsegs = rsegs.astype(np.int32)
    csegs = csegs.astype(np.int32)

    inputs = []
    flowMap = {}
    for i in range(nr):
        for j in range(nc):
            cp = opts.copy()
            cp['rseg'] = rsegs[i,j]
            cp['cseg'] = csegs[i,j]
            cp['i'] = i 
            cp['j'] = j 
            cp['flowMap'] = flowMap
            inputs.append(cp)

    # execute first job to load flow into memory
    tic()
    results = [trainModel(inputs[0])]
    toc()
    # tic()
    # results += [trainModel(inputs[1])]
    # toc()
    # exit(0)

    pool = ThreadPool(opts['numthread'])
    print('Using {} threads working on {} parallel jobs ...'.format(opts['numthread'], len(inputs)))
    tic()
    results += pool.map(trainModel, inputs[1:], 1)
    # results = pool.map(foo, inputs, 1)
    pool.close()
    pool.join()
    toc()

    vlmses = np.empty_like(rsegs, dtype=np.float32)
    vlvars = np.empty_like(rsegs, dtype=np.float32)
    agmses = np.empty_like(rsegs, dtype=np.float32)
    agvars = np.empty_like(rsegs, dtype=np.float32)
    for i, res in enumerate(results):
        inp = inputs[i]
        i = inp['i']
        j = inp['j']
        vlmse, vlvar, agmse, agvar = res
        vlmses[i,j] = vlmse
        vlvars[i,j] = vlvar
        agmses[i,j] = agmse
        agvars[i,j] = agvar
    pickle.dump(rsegs , open('{0}/{1}.p'.format(SCRATCH_PATH, "rsegs"), "wb" ))
    pickle.dump(csegs , open('{0}/{1}.p'.format(SCRATCH_PATH, "csegs"), "wb" ))
    pickle.dump(vlmses  , open('{0}/{1}.p'.format(SCRATCH_PATH, "vlmses" ), "wb" ))
    pickle.dump(vlvars  , open('{0}/{1}.p'.format(SCRATCH_PATH, "vlvars" ), "wb" ))
    pickle.dump(agmses  , open('{0}/{1}.p'.format(SCRATCH_PATH, "agmses" ), "wb" ))
    pickle.dump(agvars  , open('{0}/{1}.p'.format(SCRATCH_PATH, "agvars" ), "wb" ))

def plot():
    rsegs = np.array(pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "rsegs.p"), "rb" )))
    csegs = np.array(pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "csegs.p"), "rb" )))
    vlmses  = np.array(pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "vlmses.p") , "rb" )))
    vlvars  = np.array(pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "vlvars.p") , "rb" )))
    agmses  = np.array(pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "agmses.p") , "rb" )))
    agvars  = np.array(pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "agvars.p") , "rb" )))

    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(1,2,1, projection='3d')
    surf = ax.plot_surface(rsegs, csegs, vlmses,
            rstride=1,           # row step size
            cstride=1,           # column step size
            cmap=plt.cm.summer,        # colour map
            linewidth=1,         # wireframe line width
            antialiased=True
            )
    ax.set_title('Velocity Mean Squared Error')
    ax.set_xlabel('# Vertical Slices')
    ax.set_ylabel('# Horizontal Slices')
    ax.set_zlabel('km/h')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # ax = fig.add_subplot(1,2,2, projection='3d')
    # surf = ax.plot_surface(rsegs, csegs, vlvars,
            # rstride=1,           # row step size
            # cstride=1,           # column step size
            # cmap=plt.cm.RdPu,        # colour map
            # linewidth=1,         # wireframe line width
            # antialiased=True
            # )
    # ax.set_title('Variance Score')
    # ax.set_xlabel('Vertical Segmentation')
    # ax.set_ylabel('Horizontal Segmentation')
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    ax = fig.add_subplot(1,2,2, projection='3d')
    surf = ax.plot_surface(rsegs, csegs, agmses,
            rstride=1,           # row step size
            cstride=1,           # column step size
            cmap=plt.cm.summer,        # colour map
            linewidth=1,         # wireframe line width
            antialiased=True
            )
    ax.set_title('Angular Velocity Mean Squared Error')
    ax.set_xlabel('# Vertical Slices')
    ax.set_ylabel('# Horizontal Slices')
    ax.set_zlabel('deg/s')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    info = 'min vlmse:{0}'.format(vlmses.min())
    rs, cs = np.where(vlmses==vlmses.min())
    for r,c in zip(rs,cs):
        info += ' at {0}'.format((rsegs[r,c],csegs[r,c]))

    info += ' min agmse:{0}'.format(agmses.min())
    rs, cs = np.where(agmses==agmses.min())
    for r,c in zip(rs,cs):
        info += ' at {0}'.format((rsegs[r,c],csegs[r,c]))

    print(info)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize a sequence of images as video')
    parser.add_argument('--path', dest='path', action='store', 
            default='{0}2011_09_26-2/data'.format(KITTI_PATH),
            help='Specify path for the image files')
    parser.add_argument('--delay', dest='delay', nargs='?', default=0.05, type=float,
            help='Amount of delay between images')
    parser.add_argument('--start-frame', dest='startframe', nargs='?', default=0, type=int,
            help='Starting frame to play')
    parser.add_argument('--end-frame', dest='endframe', nargs='?', default=-1, type=int,
            help='Ending frame to play, -1 for last frame')
    parser.add_argument('--num-frame', dest='numframe', nargs='?', default=-1, type=int,
            help='Number of frame to play, -1 for all frames')
    parser.add_argument('--mode', dest='mode', action='store', default='plot')
    parser.add_argument('--rseg', dest='rseg', nargs='?', default=3, type=int,
            help='Number of vertical segmentation in computing averaged flow')
    parser.add_argument('--cseg', dest='cseg', nargs='?', default=4, type=int,
            help='Number of horizontal segmentation in computing averaged flow')
    parser.add_argument('--num-thread', dest='numthread', nargs='?', default=4, type=int,
            help='number of thread to run training')

    (opts, args) = parser.parse_known_args()

    if (opts.mode=='train'):
        exp(vars(opts))
    elif (opts.mode=='plot'):
        plot()

if __name__ == "__main__":
    main()
