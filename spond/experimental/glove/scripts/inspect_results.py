import numpy as np
import os
import pandas as pd
import sys


import socket
if socket.gethostname().endswith('pals.ucl.ac.uk'):
    # set up data path
    resultspath = "/home/petra/spond/spond/experimental/glove/results"
    datapath = "/home/petra/data"
else:
    1/0

from spond.experimental.openimages.readfile import readlabels

all_labels_file = os.path.join(datapath, 'all_labels.csv')

# labels = machine IDs to index
# names = machine IDs to names
all_labels, all_names = readlabels(all_labels_file, rootdir=None)

name_to_label = {v: k for k, v in all_names.items()}
index_to_label = {v: k for k, v in all_labels.items()}
index_to_name = {v: all_names[k] for k, v in all_labels.items()}
name_to_index = {v: k for k, v in index_to_name.items()}

# each of the labels files is just a big list of machine IDs and their
# corresponding names
datafiles = {
    'openimages': {
        'labels':  os.path.join(datapath, 'openimages', 'oidv6-class-descriptions.csv'),
    },
    'audioset': {
        'labels':  os.path.join(datapath, 'audioset', 'class_labels.csv'),
    },
}

lookup = {
    'openimages': {}, 'audioset': {},
}


tags = ("openimages", "audioset")


for tag in tags:
    labelsfn = datafiles[tag]['labels']
    included_labels = pd.read_csv(os.path.join(datapath, tag, labelsfn),
                                  header=0)
    datafiles[tag]['included_labels'] = included_labels
    # for consistency. The openimages one has different titles
    included_labels.columns = ['mid', 'display_name']
    keep = np.array([all_labels[label] for label in included_labels['mid'].values])
    lookup[tag]['included_index'] = keep
    lookup[tag]['included_names'] = [index_to_name[ind] for ind in keep]
    # we also need the index of the name in this domain
    lookup[tag]['name_to_index'] = {
        dd['display_name']: idx for idx, dd in included_labels.iterrows()
    }

# now find the labels that are in both domains, and the indexes of those labels
# in the embeddings.
union = [
    item for item in lookup['audioset']['included_names']
    if item in lookup['openimages']['included_names']
]

for tag in tags:
    lookup[tag]['union'] = {
        name: lookup[tag]['name_to_index'][name] for name in union
    }


stores = {
    tag: pd.HDFStore(os.path.join(resultspath, tag, "ProbabilisticGlove", f"{tag}_analytics.hdf5"), 'r')
    for tag in tags
}

corrs = {
    tag: stores[tag]['mostalike_correlation']
    for tag in tags
}

dists = {
    tag: stores[tag]['mostalike_distance']
    for tag in tags
}

def mostalike(tag, concept, seeds=[1,2,3,4,5]):
    for seed in seeds:
        print(f"Seed: {seed}")
        print(corrs[tag][seed][concept])
        print(dists[tag][seed][concept])
