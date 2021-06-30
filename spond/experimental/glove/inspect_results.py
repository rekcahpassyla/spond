import os
import pandas as pd

from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove

import socket
if socket.gethostname().endswith('pals.ucl.ac.uk'):
    # set up pythonpath
    ppath = '/home/petra/spond'
    # set up data path
    datapath = '/home/petra/data'
else:
    1/0

datapath = "/home/petra/spond/spond/experimental/glove/results"

tags = ("openimages", "audioset")

stores = {
    tag: pd.HDFStore(os.path.join(datapath, tag, "ProbabilisticGlove", f"{tag}_analytics.hdf5"), 'r')
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
