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

datafiles = {
    'openimages': {
        'all_labels':  os.path.join(datapath, 'openimages', 'oidv6-class-descriptions.csv'),
    },
    'audioset': {
        'all_labels':  os.path.join(datapath, 'audioset', 'all_labels.csv'),
        'included_labels': pd.read_csv(
            os.path.join('/home/petra/data', 'audioset', 'class_labels_indices.csv'),
            index_col=0
        )
    },
}

lookup = {
    'openimages': {}, 'audioset': {},
}    


tags = ("openimages", "audioset")


for tag in tags:
    labelsfn = datafiles[tag]['all_labels']
    # labels = machine IDs to index
    # names = category name to index
    labels, names = readlabels(labelsfn, rootdir=None)
    name_to_label = {v: k for k, v in names.items()}
    index_to_label = {v: k for k, v in labels.items()}
    index_to_name = {v: names[k] for k, v in labels.items()}
    name_to_index = {v: k for k, v in index_to_name.items()}
    if tag == 'openimages':
        datafiles[tag]['included_labels'] = pd.DataFrame({
            'mid': pd.Series(index_to_label),
            'display_name': pd.Series(index_to_name)
        })
    
    keep = np.array([labels[label] for label in datafiles[tag]['included_labels']['mid'].values])
    # index of label in file to index in the output embedding
    lookup[tag]['label_to_index'] = labels
    lookup[tag]['name_to_index'] = name_to_index
    lookup[tag]['name_to_label'] = name_to_label
    # the audioset embeddings are for all labels,
    # so use the 'keep' array to further filter only the ones we want. 
    # for openimages it has no effect
    lookup[tag]['included_index'] = keep
    lookup[tag]['included_names'] = [index_to_name[ind] for ind in keep]

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
