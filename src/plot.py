from os import listdir
from os.path import isfile, isdir, join, splitext, basename, dirname
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.font_manager import FontProperties
from functools import partial

def parse(**options):
    path = options['path']
    # parse logs
    global results, logs
    results = {}
    logs = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.txt')]

    for log in logs:
        #print('Parsing {}'.format(log))
        with open(join(path, log), 'r') as f:
            results[log] = {}
            results[log]['epoch'] = []
            results[log]['train_mse'] = []
            results[log]['val_mse'] = []
            for line in f:
                if 'Configuration' in line:
                    key, val = line.split('Configuration')[1].split(' ')[1].split('=')
                    results[log][key] = val.replace('\n', '')
                if 'Error' in line or 'Fail' in line or 'Interrupt' in line or 'Killed' in line: # broken log
                    results.pop(log, None)
                    continue
                if 'Overall train mse' in line:
                    try:
                      results[log]['epoch'].append(int(line.split('Epoch ')[1].split(',')[0]))
                      results[log]['train_mse'].append(float(line.split('Overall train mse = ')[1].split(',')[0]))
                      results[log]['val_mse'].append(float(line.split('Overall val mse = ')[1]))
                    except:
                      continue
        k = 'valmode';
        if k not in results[log]: results[log][k] = 0

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
    label += '_lr{}'.format(d['learning_rate'])
    # label += '_'
    # label += '#{}'.format(d['num_frames'])

    return label

def linestyle(d):
    if int(d['convmode'])==0: return '--'
    elif int(d['convmode'])==1: return '-.'
    elif int(d['convmode'])==2: return '-'

def color(d):
    if int(d['speedmode'])==0: return 'r' 
    elif int(d['speedmode'])==1: return 'g'
    elif int(d['speedmode'])==2: return 'b'
    elif int(d['speedmode'])==3: return 'royalblue'
    elif int(d['speedmode'])==4: return 'c'

def to_lookup(log):
    return lookup(results[log])

def to_key(log, key):
    return results[log][key]

def sort_logs(logs, key=None):
    if key is None:
        return sorted(logs, key = to_lookup)
    else:
        return sorted(logs, key = partial(to_key, key=key))

def plot_all_loss(**options):
    logscale = options['logscale']
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,2.5), sharey=True)
    handles = []
    for log in logs:
        ls = linestyle(results[log])
        cl = color(results[log])
        label = lookup(results[log])
        if logscale:
            handles += (ax1.semilogy(results[log]['epoch'], results[log]['train_mse'], label=label,
                color=cl, linestyle=ls))
        else:
            handles += (ax1.plot(results[log]['epoch'], results[log]['train_mse'], label=label,
                color=cl, linestyle=ls))
    ax1.set_title('train_loss')
    ax1.grid(True, ls='dashed')
    ax1.set_xlabel('epoch')
    handles = []
    slogs = sort_logs(logs)
    for log in slogs:
        ls = linestyle(results[log])
        cl = color(results[log])
        label = lookup(results[log])
        if logscale:
            handles += (ax2.semilogy(results[log]['epoch'], results[log]['val_mse'], label=label,
                color=cl, linestyle=ls))
        else:
            handles += (ax2.plot(results[log]['epoch'], results[log]['val_mse'], label=label,
                color=cl, linestyle=ls))
    ax2.set_ylim([0,max(results[log]['val_mse'])+20])
    ax2.set_title('val_loss')
    ax2.grid(True, ls='dashed')
    ax2.set_xlabel('epoch')
    ax2.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.85, right=0.7, left=0.05)
    plt.savefig('{}/result_all.png'.format(options['path']))

def plot_sep_loss(**options):
    plt.clf()
    for i, log in enumerate(logs):
        fig, ax1 = plt.subplots(figsize=(4,3))
        handles = []
        if logscale:
            handles += (ax1.semilogy(results[log]['epoch'], results[log]['train_mse'], label='train_mse'))
            handles += (ax1.semilogy(results[log]['epoch'], results[log]['val_mse'], label='val_mse'))
        else:
            handles += (ax1.plot(results[log]['epoch'], results[log]['train_mse'], label='train_mse'))
            handles += (ax1.plot(results[log]['epoch'], results[log]['val_mse'], label='val_mse'))
        ax1.set_title('loss of {}'.format(lookup(results[log])))
        ax1.grid(True, ls='dashed')
        ax1.set_xlabel('epoch')
        ax1.legend(handles=handles)
        plt.gcf().subplots_adjust(bottom=0.15, top=0.75)
        plt.savefig('{}/result_{}.png'.format(options['path'], log.replace('result_','').replace('.txt','')))

def plot_param_sweep(mode, **options):
    logscale = options['logscale']
    plt.clf()
    fig, ax = plt.subplots(figsize=(4,3))

    logs_filtered = list(filter(lambda log: 'lr_' in log, logs))
    logs_filtered.sort(key=lambda log : float(results[log][mode]))

    x = list(map(lambda log: float(results[log][mode]), logs_filtered))
    y_train = list(map(lambda log: results[log]['train_mse'][-1], logs_filtered))
    y_val = list(map(lambda log: results[log]['val_mse'][-1], logs_filtered))

    ax.plot(x, y_train, 'b', label='train_mse')
    ax.plot(x, y_val, 'b--', label='val_mse')
    
    # ax.set_title(mode + ' Sweep')
    ax.grid(True, ls='dashed')
    # ax.set_xlabel(mode)
    ax.set_xscale('log')
    #ax.set_xlim(-1, float(max(x))+0.05)
    ax.legend(loc='best')
    plt.savefig('{}/param_tuning_lr_{}.png'.format(options['path'], mode))

def plot_downsample(**options):
    fig, axes = plt.subplots(2,1, figsize=(4.5,6))
    convmodes = [0, 1]
    neq = ['result_20170609020814.txt', 'result_20170609020824.txt']
    thresh = [20, 10]
    for i, convmode in enumerate(convmodes):
        options['toshow'] = \
            "speedmode=0,flowmode=2,dropout=0.5,learning_rate=0.0001,convmode={},name!={}".format(convmode,
                    neq[i])
        filtered = filterby(**options)
        logs_filtered = [log for log, info in filtered[True]]
        options['toshow'] = \
            "speedmode=0,flowmode=1,dropout=0.5,learning_rate=0.0001,convmode={},train_mse<{}".format(convmode,
                    thresh[i])
        filtered = filterby(**options)
        logs_filtered += [log for log, info in filtered[True]] 
        x = []
        x_label = []
        for log in logs_filtered:
            if int(results[log]['flowmode'])==2:
                x.append(int(results[log]['rseg']) * int(results[log]['cseg']))
                x_label.append('({},{})'.format(int(results[log]['rseg']),
                    int(results[log]['cseg'])))
            else:
                x.append(1242*375)
                x_label.append('(1242,375)')
        y_train = list(map(lambda log: results[log]['train_mse'][-1], logs_filtered))
        y_val = list(map(lambda log: results[log]['val_mse'][-1], logs_filtered))
        sort = sorted(zip(x, y_train, y_val))
        x = [xx for xx ,yt, yv in sort]
        y_train = [yt for xx ,yt, yv in sort]
        y_val = [yv for xx ,yt, yv in sort]
        axes[i].plot(x, y_train, 'go-', label='train_mse')
        axes[i].plot(x, y_val, 'ro--', label='val_mse')
        axes[i].legend(loc='best')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(x_label, rotation=30)
        axes[i].grid(True, ls='dashed')
        axes[i].set_title(lookup(results[log]))
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig('{}/result_downsample.png'.format(options['path']))

def plot_batch_size(**options):
    fig, ax = plt.subplots(figsize=(4,3))
    batches = {}
    for i, log in enumerate(logs):
        if 'batch_size' in results[log]:
            bs = results[log]['batch_size']
            train_mse = results[log]['train_mse'][-1]
            if (bs, 'train') not in batches:
                batches[(bs, 'train')] = [train_mse]
            else:
                batches[(bs, 'train')].append(train_mse)
            val_mse = results[log]['val_mse'][-1]
            if (bs, 'val') not in batches:
                batches[(bs, 'val')] = [val_mse]
            else:
                batches[(bs, 'val')].append(val_mse)
    keys = [bs for bs in batches]
    data = [batches[bs] for bs in keys]
    label = [bs+'_'+tp for bs, tp in keys]
    # multiple box plots on one figure
    plt.boxplot(data, 0, 'gD')
    plt.xticks(range(1, len(data)+1), label)
    ax.set_ylabel('Averaged MSE Loss')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('{}/result_batch.png'.format(options['path']))

def plot_valsample(**options):
    options['toshow'] = "valmode=0"
    filtered = filterby(**options)
    oldlogs = [log for log,info in filtered[True]]
    newlogs = [log for log,info in filtered[False]]
    oldx = [results[log]['train_mse'][-1] for log in oldlogs]
    oldy = [results[log]['val_mse'][-1] for log in oldlogs]
    newx = [results[log]['train_mse'][-1] for log in newlogs]
    newy = [results[log]['val_mse'][-1] for log in newlogs]
    fig, ax = plt.subplots(figsize=(4,3))
    plt.plot(oldx, oldy, 'ro')
    plt.plot(newx, newy, 'go')
    ax.grid(True, ls='dashed')
    mx = max(max(oldx), max(newx), max(oldy), max(newy))+1
    mx = 100 
    ax.set_xlim([0, mx])
    ax.set_ylim([0, mx])
    ax.set_aspect('equal')
    ax.set_xlabel('Train MSE Loss')
    ax.set_ylabel('Val MSE Loss')
    plt.tight_layout()
    plt.savefig('{}/result_valsample.png'.format(options['path']))

def printc(txt, c):
    if c=="g":
        print('\033[92m' + txt + '\033[0m')

def filterby(**options):
    toshows = options['toshow']
    toshows = toshows.split(',')
    filtered = {True: [], False: []}
    for i, log in enumerate(sort_logs(logs, 'convmode')):
        info = log
        val_mse = None
        if len(results[log]['epoch']) > 0:
            info += ' epoch={}'.format(results[log]['epoch'][-1])
        if len(results[log]['train_mse']) > 0:
            info += ' train_mse={}'.format(results[log]['train_mse'][-1])
        if len(results[log]['val_mse']) > 0:
            val_mse = results[log]['val_mse'][-1]
            info += ' val_mse={}'.format(val_mse)
        info += ' {}'.format(lookup(results[log]))
        # if 'weight_init' in results[log]:
            # info += ' {}'.format(results[log]['weight_init'])
        k='decay_rate'; info += ' {}={}'.format(k, results[log][k])
        k='decay_step'; info += ' {}={}'.format(k, results[log][k])
        k='dropout'; info += ' {}={}'.format(k, results[log][k])
        if 'flowmode' in results[log]:
            k='flowmode'; info += ' {}={}'.format(k, results[log][k])
        k='rseg'; info += ' {}={}'.format(k, results[log][k])
        k='cseg'; info += ' {}={}'.format(k, results[log][k])
        cond = True
        for toshow in toshows:
            if '!=' in toshow:
                key, val = toshow.split('!=')
                if key in ['val_mse', 'train_mse', 'epoch']:
                    cond &= key in results[log] and float(results[log][key][-1]) != float(val)
                elif key in ['name']:
                    cond &= val != log
                else:
                    cond &= key in results[log] and float(results[log][key]) != float(val)
            elif '=' in toshow:
                key, val = toshow.split('=')
                if key in ['val_mse', 'train_mse', 'epoch']:
                    cond &= key in results[log] and float(results[log][key][-1]) == float(val)
                elif key in ['name']:
                    cond &= val in log
                else:
                    cond &= key in results[log] and float(results[log][key]) == float(val)
            elif '<' in toshow:
                key, val = toshow.split('<')
                if key in ['val_mse', 'train_mse', 'epoch']:
                    cond &= key in results[log] and len(results[log][key])>0 and float(results[log][key][-1]) < float(val)
                else:
                    cond &= key in results[log] and float(results[log][key]) < float(val)
            else:
                if toshow in results[log] and toshow not in info:
                    k=toshow; info += ' {}={}'.format(k, results[log][k])
        if cond and len(toshows)>0:
            filtered[True].append((log, info))
        else:
            filtered[False].append((log, info))
    return filtered

def show(**options):
    print('\nShowing configuration ...')
    filtered = filterby(**options)
    for log, info in filtered[False]:
        print(info)
    for log, info in filtered[True]:
        printc(info, 'g')

def main():
    usage = "Usage: plot [options --path]"
    parser = argparse.ArgumentParser(description='Visualize a sequence of images as video')
    parser.add_argument('--path', dest='path', action='store', default='../results',
            help='Specify path for result logs')
    parser.add_argument('--logscale', dest='logscale', action='store_true',default=False,
        help='Use log scale')
    parser.add_argument('--show', dest='toshow', action='store', default='None',
            help='Specify condition to highlight. e.g. --show "val_mse<3" ')
    parser.add_argument('--plot', dest='toplot', action='store', default='None',
            help='Specify plots to generate. e.g. --plot "all_loss sep_loss learning_rate dropout \
            batch_size, downsample" ')
    (options, args) = parser.parse_known_args()

    options = vars(options)
    parse(**options)

    if options['toshow'] != 'None':
        show(**options)
    
    if options['toplot'] != 'None':
        for plot in options['toplot'].split(' '):
            if plot == 'all_loss':
                plot_all_loss(**options)
            elif plot == 'sep_loss':
                plot_sep_loss(**options)
            elif plot == 'learning_rate':
                plot_param_sweep(plot, **options)
            elif plot == 'dropout':
                plot_param_sweep(plot, **options)
            elif plot == 'batch_size':
                plot_batch_size(**options)
            elif plot == 'downsample':
                plot_downsample(**options)
            elif plot == 'valsample':
                plot_valsample(**options)

if __name__ == "__main__":
    main()
