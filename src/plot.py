from os import listdir
from os.path import isfile, isdir, join, splitext, basename, dirname
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse(**options):
    path = options['path']
    # parse logs
    global results, logs
    results = {}
    logs = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.txt')]

    for log in logs:
        with open(join(path, log), 'r') as f:
            results[log] = {}
            results[log]['epoch'] = []
            results[log]['train_mse'] = []
            results[log]['val_mse'] = []
            for line in f:
                if 'Configuration' in line:
                    key, val = line.split(' ')[1].split('=')
                    results[log][key] = val
                if 'Error' in line or 'Fail' in line: # broken log
                    print('broken log {}'.format(log))
                    results.pop(log, None)
                    continue
                if 'Overall train mse' in line:
                    results[log]['epoch'].append(int(line.split('Epoch ')[1].split(',')[0]))
                    results[log]['train_mse'].append(float(line.split('Overall train mse = ')[1].split(',')[0]))
                    results[log]['val_mse'].append(float(line.split('Overall val mse = ')[1]))

def lookup(d):
    label = ''
    if int(d['speedmode'])==0: label += 'flow'
    elif int(d['speedmode'])==1: label += 'flow+obj'
    elif int(d['speedmode'])==2: label += 'flow+rgb'
    elif int(d['speedmode'])==3: label += 'flow+obj+rgb'
    elif int(d['speedmode'])==4: label += 'rgb'
    label += '_'
    if int(d['convmode'])==0: label += 'baseline-cnn'
    elif int(d['convmode'])==1: label += 'resnet'
    elif int(d['convmode'])==2: label += 'alexnet'
    label += '_'
    label += '#{}'.format(d['num_frames'])

    return label

def plot_all_loss(**options):
    plt.figure(1, figsize=(8,3))
    ax1 = plt.subplot(1,2,1)
    handles = []
    for log in logs:
        label = lookup(results[log])
        handles += (ax1.plot(results[log]['epoch'], results[log]['train_mse'], label=label))
    ax1.set_title('train_loss')
    ax1.grid(True)
    ax1.set_xlabel('epoch')
    ax2 = plt.subplot(1,2,2)
    handles = []
    for log in logs:
        label = lookup(results[log])
        handles += (ax2.plot(results[log]['epoch'], results[log]['val_mse'], label=label))
    ax2.set_title('val_loss')
    ax2.grid(True)
    ax2.set_xlabel('epoch')
    ax2.legend(handles=handles)
    plt.gcf().subplots_adjust(bottom=0.15, top=0.85)
    plt.savefig('{}/loss_all.png'.format(options['path']))

def plot_sep_loss(**options):
    plt.clf()
    for i, log in enumerate(logs):
        plt.figure(i+1, figsize=(4,3))
        handles = []
        ax1 = plt.subplot()
        handles += (ax1.plot(results[log]['epoch'], results[log]['train_mse'], label='train_mse'))
        handles += (ax1.plot(results[log]['epoch'], results[log]['val_mse'], label='val_mse'))
        ax1.set_title('loss of {}'.format(lookup(results[log])))
        ax1.grid(True)
        ax1.set_xlabel('epoch')
        ax1.legend(handles=handles)
        plt.gcf().subplots_adjust(bottom=0.15, top=0.75)
        plt.savefig('{}/loss_{}.png'.format(options['path'], log.replace('result_','').replace('.txt','')))

def main():
    usage = "Usage: plot [options --path]"
    parser = argparse.ArgumentParser(description='Visualize a sequence of images as video')
    parser.add_argument('--path', dest='path', action='store', default='../results',
            help='Specify path for result logs')
    (options, args) = parser.parse_known_args()

    options = vars(options)
    parse(**options)
    plot_all_loss(**options)
    plot_sep_loss(**options)

if __name__ == "__main__":
    main()
