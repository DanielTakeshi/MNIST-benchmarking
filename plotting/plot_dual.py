"""
Plotting for the dual MNIST case.
(c) 2018 by Daniel Seita
"""
import argparse, os, pickle, sys, matplotlib, names
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(edgeitems=100, linewidth=100, suppress=True)
from collections import defaultdict

# Some matplotlib settings.
plt.style.use('seaborn-darkgrid')
error_region_alpha = 0.25
title_size = 22
tick_size = 18
legend_size = 19
xsize, ysize = 19, 19
lw, ms = 3, 8
COLORS = ['red', 'blue', 'yellow', 'brown', 'purple']


def get_info(dirs):
    """
    Iterate through the directories in some group. For each, append the
    attribute list to the default dict list. Thus at the end, the info
    dictionary has a bunch of lists, each of which are length 5 (if there were 5
    directories, for instance) and thus consist of 5 lists. Then convert these
    to numpy arrays, so that the shapes of all elements in the info dictionary
    are (5,num_time_steps).
    """
    info = defaultdict(list)

    for dd in dirs:
        dd = 'logs/'+dd
        A = np.genfromtxt(
                os.path.join(dd, 'log.txt'), delimiter='\t', dtype=None, names=True
        )
        info['TrainEpochs'].append(A['TrainEpochs'])
        for attr in names.ATTRIBUTES:
            if attr not in A.dtype.names:
                continue
            info[attr].append(A[attr])
    target_shape = np.array(info['TrainEpochs']).shape

    # Turn everything to numpy arrays so we can take means and stdevs.
    for attr in names.ATTRIBUTES:
        if attr not in info:
            continue
        info[attr] = np.array(info[attr])
        assert info[attr].shape == target_shape
        #print("info[{}].shape: {}".format(attr, info[attr].shape))
    return info


def plot(all_info, figname):
    """ Plots default (1) and then ss (2) together.

    A bit tricky because we also need to ensure that the trajectories last the
    same number of steps. Fortunately the latest code additions should make this
    happen...
    """
    # List of dictionaries, makes things easier.
    infos = []
    for dirs in all_info['directories']:
        infos.append(get_info(dirs))
    nrows = int( (len(names.ATTRIBUTES)+1)/2 )
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22,6*nrows))
    past = 20

    # Iterate through the directories, makes things easier.
    for idx,(info, nn, cc) in enumerate(zip(infos, all_info['names'], COLORS)):
        xc = (infos[idx])['TrainEpochs'][0]
        print("Now plotting name {} ...".format(nn))
        print("note that xc.shape: {}".format(xc.shape))
        attr_idx = 0
        for r in range(nrows):
            for c in range(ncols):
                attr = names.ATTRIBUTES[attr_idx]
                attr_idx += 1
                if attr not in info.keys():
                    continue
                print("\tattr {}".format(attr))
                data = info[attr]
                data_mean = np.mean(data, axis=0)
                data_std  = np.std(data, axis=0)
                label = "{}; last{}_avg({:.3f})".format(nn, past, np.mean(data_mean[-past:]))
                axes[r,c].fill_between(xc,
                                       data_mean-data_std,
                                       data_mean+data_std,
                                       alpha=error_region_alpha,
                                       facecolor=cc)
                axes[r,c].plot(xc, data_mean, '-', lw=lw, color=cc, label=label)

    # Bells and whistles
    attr_idx = 0
    for r in range(nrows):
        for c in range(ncols):
            attr = names.ATTRIBUTES[attr_idx]
            attr_idx += 1
            axes[r,c].tick_params(axis='x', labelsize=tick_size)
            axes[r,c].tick_params(axis='y', labelsize=tick_size)
            axes[r,c].set_xlabel("Training Epochs", fontsize=xsize)
            axes[r,c].set_ylabel("{}".format(attr), fontsize=ysize)
            if 'acc' in attr.lower():
                axes[r,c].set_ylim([0.0, 1.0])
            elif 'diff' in attr.lower():
                axes[r,c].set_ylim([0.0, 2.0])
            axes[r,c].set_title("{}".format(attr), fontsize=title_size)
            axes[r,c].legend(loc='best', ncol=1, prop={'size':legend_size})
    plt.tight_layout()
    plt.savefig(figname)
    print("Just saved figure: {}".format(figname))


if __name__ == "__main__":
    #plot(names.MAR07_FIRST_TRY, 'figures/mar07_first_try.png')
    #plot(names.MAR07_1000_EPOCHS, 'figures/mar07_1000_epochs.png')
    #plot(names.MAR12_3000_EPOCHS, 'figures/mar12_3000_epochs.png')
    plot(names.MAR28_SMALL_DATA, 'figures/mar28_small_data.png')
