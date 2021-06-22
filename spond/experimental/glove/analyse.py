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
    tag = 'openimages'
    gpu = True
    labelsfn = os.path.join(datapath, tag, 'oidv6-class-descriptions.csv')
    resultspath = 'results'
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
seeds = (1, 2, 3, 4, 5)

N = 200

if tag == 'audioset':
    # now we need to find the indexes of the audio labels in the all-labels file
    included_labels = pd.read_csv("/opt/github.com/spond/spond/experimental/audioset/class_labels_indices.csv",
                                  index_col=0)
else:
    included_labels = pd.DataFrame({
        'mid': pd.Series(index_to_label),
        'display_name': pd.Series(index_to_name)
    })

corrs = {}



for seed in seeds:
    df = s[str(seed)]
    corrs[seed] = np.corrcoef(df.values)
    plt.figure()
    fig = plt.imshow(corrs[seed])
    plt.colorbar(fig)
    plt.title(f'Correlation of dot product similarity, {seed}')
    plt.savefig(os.path.join(rdir, f'{tag}_dotsim_corr_{seed}.png'))

# now work out cross correlations

crosscorrs = {}

for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i+1:]:
        c1 = s[str(seed1)].values.ravel()
        c2 = s[str(seed2)].values.ravel()
        crosscorrs[(seed1, seed2)] = np.corrcoef(c1, c2)[0][1]

s.close()

keep = np.array([labels[label] for label in included_labels['mid'].values])

from probabilistic_glove import ProbabilisticGlove
import os


entropies = {}
models = {}
for seed in seeds:

    model = ProbabilisticGlove.load(os.path.join(rdir, f'{tag}_ProbabilisticGlove_{seed}.pt'))
    models[seed] = model
    cc = np.corrcoef(model.glove_layer.wi_mu.weight.detach()[keep].numpy())
    cc = np.abs(cc)
    ccmax = cc.copy()
    ccmax[np.isclose(cc, 1)] = -np.inf

    ccmin = cc.copy()
    ccmin[np.isclose(cc, 1)] = np.inf

    # display the label with the highest correlation to a particular label
    maxes = {
        included_labels['display_name'][i]: (included_labels['display_name'][idx], cc[i][idx])
        for i, idx in enumerate(ccmax.argmax(axis=0))
    }
    # calculate entropy
    ent = model.glove_layer.entropy().detach()[keep]
    # sort it
    ents, indices = ent.sort()
    ordered_labels = [included_labels['display_name'][item] for item in indices.numpy()]
    entropies[seed] = pd.Series(
        data=ents.numpy(), index=ordered_labels
    )

# do the same for lowest
#mins = {
#    index_to_name[start + i]: (index_to_name[start + offset], cc[i][offset])
#    for i, offset in enumerate(ccmin.argmin(axis=0))
#}

