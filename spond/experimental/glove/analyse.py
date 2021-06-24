import gc
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


import socket
if socket.gethostname().endswith('pals.ucl.ac.uk'):
    # set up pythonpath
    ppath = '/home/petra/spond'
    # set up data pth
    datapath = '/home/petra/data'
    gpu = True
    #tag = 'audioset'
    #labelsfn = os.path.join(datapath, tag, 'all_labels.csv')
    tag = 'openimages'
    labelsfn = os.path.join(datapath, tag, 'oidv6-class-descriptions.csv')
    resultspath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results')
else:
    ppath = '/opt/github.com/spond/spond/experimental'
    datapath = ppath
    gpu = False
    tag = 'audioset'
    labelsfn = "/opt/github.com/spond/spond/experimental/audioset/all_labels.csv"
    resultspath = '/opt/github.com/spond/spond/experimental/glove/results/'

sys.path.append(ppath)

from spond.experimental.openimage.readfile import readlabels
rdir = os.path.join(resultspath, f'{tag}/ProbabilisticGlove')


labels, names = readlabels(labelsfn, rootdir=None)

name_to_label = {v: k for k, v in names.items()}
index_to_label = {v: k for k, v in labels.items()}
index_to_name = {v: names[k] for k, v in labels.items()}

s = pd.HDFStore(os.path.join(rdir, f'{tag}_means_dot.hdf5'), 'r')
outfile = pd.HDFStore(os.path.join(rdir, f'{tag}_analytics.hdf5'))
seeds = (1, 2, 3, 4, 5)

N = 200

if tag == 'audioset':
    # now we need to find the indexes of the audio labels in the all-labels file
    #included_labels = pd.read_csv("/opt/github.com/spond/spond/experimental/audioset/class_labels_indices.csv",
    #                              index_col=0)
    included_labels = pd.read_csv(
        os.path.join(datapath, tag, 'class_labels_indices.csv'),
        index_col=0)
else:
    included_labels = pd.DataFrame({
        'mid': pd.Series(index_to_label),
        'display_name': pd.Series(index_to_name)
    })


for seed in seeds:
    df = s[str(seed)]
    corrs = np.corrcoef(df.values)
    plt.figure()
    fig = plt.imshow(corrs)
    plt.colorbar(fig)
    plt.title(f'Correlation of dot product similarity, {seed}')
    plt.savefig(os.path.join(rdir, f'{tag}_dotsim_corr_{seed}.png'))
    del corrs
    gc.collect()

# now work out cross correlations

crosscorrs = {}

for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i+1:]:
        c1 = s[str(seed1)].values.ravel()
        c2 = s[str(seed2)].values.ravel()
        crosscorrs[(seed1, seed2)] = np.corrcoef(c1, c2)[0][1]
        del c1
        del c2
        gc.collect()

s.close()

crosscorrs = pd.Series(crosscorrs)
outfile['crosscorrs'] = crosscorrs
outfile.flush()

del crosscorrs
gc.collect()

keep = np.array([labels[label] for label in included_labels['mid'].values])

from probabilistic_glove import ProbabilisticGlove
import os


entropies = {}
models = {}
maxcorrs = {}
mincorrs = {}
for seed in seeds:
    model = ProbabilisticGlove.load(os.path.join(rdir, f'{tag}_ProbabilisticGlove_{seed}.pt'))
    models[seed] = model
    cc = np.corrcoef(model.glove_layer.wi_mu.weight.detach()[keep].numpy())
    cc = np.abs(cc)
    # replace values of 1 with inf or -inf so that we can sort easily
    ccmax = cc.copy()
    ccmax[np.isclose(cc, 1)] = -np.inf

    ccmin = cc.copy()
    ccmin[np.isclose(cc, 1)] = np.inf

    # topcorrs are the indexes of the highest correlations sorted by column
    topcorrs = ccmax.argsort(axis=0)[-5:]
    bottomcorrs = ccmin.argsort(axis=0)[:5]

    # make into data structures
    maxes = pd.Series({
        included_labels['display_name'][i]:
            pd.Series(index=included_labels['display_name'][topcorrs[::-1][:,i]].values, data=ccmax[i][topcorrs[::-1][:,i]])
        for i in range(ccmax.shape[0])
    })

    mins = pd.Series({
        included_labels['display_name'][i]:
            pd.Series(index=included_labels['display_name'][bottomcorrs[:,i]].values, data=ccmin[i][bottomcorrs[:,i]])
        for i in range(ccmin.shape[0])
    })

    maxcorrs[seed] = maxes
    mincorrs[seed] = mins
    # calculate entropy
    ent = model.glove_layer.entropy().detach()[keep]
    # sort it
    ents, indices = ent.sort()
    ordered_labels = [included_labels['display_name'][item] for item in indices.numpy()]
    entropies[seed] = pd.Series(
        data=ents.numpy(), index=ordered_labels
    )

maxcorrs = pd.DataFrame(maxcorrs)
mincorrs = pd.DataFrame(mincorrs)
entropies = pd.DataFrame(entropies)

outfile['maxcorrs'] = maxcorrs
outfile['mincorrs'] = mincorrs
outfile['entropies'] = entropies

outfile.close()

