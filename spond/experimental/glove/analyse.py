import gc
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
from scipy.spatial.distance import cdist


import socket
if socket.gethostname().endswith('pals.ucl.ac.uk'):
    # set up pythonpath
    ppath = '/home/petra/spond'
    # set up data path
    datapath = '/home/petra/data'
    gpu = True
    #tag = 'audioset'
    #labelsfn = os.path.join(datapath, tag, 'all_labels.csv')
    #train_cooccurrence_file = os.path.join(datapath, tag, 'co_occurrence_audio_all.pt')
    tag = 'openimages'
    labelsfn = os.path.join(datapath, tag, 'oidv6-class-descriptions.csv')
    train_cooccurrence_file = os.path.join(datapath, tag, 'co_occurrence.pt')
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
    train_cooccurrence_file = ''

device = torch.device("cuda:0" if gpu else "cpu")

sys.path.append(ppath)

# Can only import after path is set above
from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove
from spond.experimental.openimages.readfile import readlabels

rdir = os.path.join(resultspath, f'{tag}/ProbabilisticGlove')


labels, names = readlabels(labelsfn, rootdir=None)

name_to_label = {v: k for k, v in names.items()}
index_to_label = {v: k for k, v in labels.items()}
index_to_name = {v: names[k] for k, v in labels.items()}
name_to_index = {v: k for k, v in index_to_name.items()}

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
    print(f"Calculating self-correlation for seed {seed}")
    corrs = np.corrcoef(df.values)
    plt.figure()
    fig = plt.imshow(corrs)
    plt.colorbar(fig)
    plt.title(f'Correlation of dot product similarity, {seed}')
    plt.savefig(os.path.join(rdir, f'{tag}_dotsim_corr_{seed}.png'))

    del fig
    gc.collect()

    # print(f"Calculating self-distance for seed {seed}")
    # dist = cdist(df.values, df.values)  # euclidean is the default
    # plt.figure()
    # fig = plt.imshow(dist)
    # plt.colorbar(fig)
    # plt.title(f'Distance of dot product similarity, {seed}')
    # plt.savefig(os.path.join(rdir, f'{tag}_dotsim_dist_{seed}.png'))
    # del fig
    del corrs
    del df
    gc.collect()


# now work out cross correlations

crosscorrs = {}

for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i+1:]:
        print(f"Calculating cross-correlation for {seed1} x {seed2}")
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


entropies = {}
models = {}
most = {}
least = {}
metric = 'distance'
for seed in seeds:
    print(f"Calculating max/min {metric}s for seed {seed}")
    model = ProbabilisticGlove.load(os.path.join(rdir, f'{tag}_ProbabilisticGlove_{seed}.pt'))
    if metric == 'correlation':
        cc = np.corrcoef(model.glove_layer.wi_mu.weight.detach()[keep].numpy())
        #cc = np.abs(cc)
        # replace values of 1 with inf or -inf so that we can sort easily
        mostlike = cc.copy()
        mostlike[np.isclose(cc, 1)] = -np.inf

        leastlike = cc.copy()
        leastlike[np.isclose(cc, 1)] = np.inf

        # top are the indexes of the highest correlations sorted by column
        # do .copy() here because later in the loop we can then delete
        # mostlike and leastlike, and free a lot of memory
        top = mostlike.argsort(axis=0)[-5:][::-1].copy()
        bottom = leastlike.argsort(axis=0)[:5].copy()
    else:
        assert metric == 'distance'
        wt = model.glove_layer.wi_mu.weight.detach()[keep]
        wt = wt.to(device)
        # compute_mode="donot_use_mm_for_euclid_dist" is required or else
        # the distance between something and itself is not 0
        # don't know why.
        dist = torch.cdist(wt, wt, compute_mode="donot_use_mm_for_euclid_dist")
        # same as above, replace values of 0 with inf or -inf so we can sort
        mostlike = dist.clone()
        mostlike[torch.isclose(dist, torch.tensor([0.0]).to(device))] = np.inf
        mostlike = mostlike.cpu().numpy()
        top = mostlike.argsort(axis=0)[:5].copy()
        del mostlike
        torch.cuda.empty_cache()
        gc.collect()

        leastlike = dist.clone()
        leastlike[torch.isclose(dist, torch.tensor([0.0]).to(device))] = -np.inf
        # top are the indexes of the lowest distances sorted by column
        # see note in the other branch about copy() and memory management
        leastlike = leastlike.cpu().numpy()
        bottom = leastlike.argsort(axis=0)[-5:][::-1].copy()
        del leastlike
        del dist
        torch.cuda.empty_cache()
        gc.collect()

    # make into data structures
    maxes = pd.Series({
        included_labels['display_name'][i]:
            pd.Series(index=included_labels['display_name'][top[:, i]].values,
                      data=mostlike[i][top[:, i]])
        for i in range(mostlike.shape[0])
    })

    mins = pd.Series({
        included_labels['display_name'][i]:
            pd.Series(index=included_labels['display_name'][bottom[:, i]].values,
                      data=leastlike[i][bottom[:, i]])
        for i in range(leastlike.shape[0])
    })

    most[seed] = maxes
    least[seed] = mins

    # calculate entropy
    ent = model.glove_layer.entropy().detach()[keep]
    # sort it
    ents, indices = ent.sort()
    ordered_labels = [included_labels['display_name'][item] for item in indices.numpy()]
    entropies[seed] = pd.Series(
        data=ents.numpy().copy(), index=ordered_labels
    )
    # delete everything not needed so we can free memory for the next loop
    del mostlike
    del leastlike
    del top
    del bottom
    del ent
    gc.collect()

most = pd.DataFrame(most)
least = pd.DataFrame(least)
entropies = pd.Series(entropies)

plt.figure()
for seed in seeds:
    plt.hist(entropies[seed].values, alpha=0.3, bins=100, label=str(seed))
plt.title(f'Entropies for {tag} per seed')
plt.legend()
plt.savefig(os.path.join(rdir, f'{tag}_entropies.png'))

outfile[f'mostalike_{metric}'] = most
outfile[f'leastalike_{metric}'] = least
outfile['entropies'] = entropies

# calculate correlations of counts with entropies, for each seed
# entropies index are alphabetical, we have to match up with the counts

cooc = torch.load(train_cooccurrence_file)
cooc = cooc.coalesce().to_dense()
counts = cooc.sum(axis=0)[keep].numpy()
counts = pd.Series(data=counts, index=[index_to_name[i] for i in keep])

entropy_count_corr = pd.Series({
    seed: np.corrcoef(
        counts.sort_index().values,
        entropies[seed].sort_index().values
    )[0][1] for seed in seeds
})

outfile['entropy_count_corr'] = entropy_count_corr

outfile.close()

