# This module will contain 2 ProbabilisticGlove layers that will be
# trained from their separate co-occurrence matrices.
# Module is not called aligner.py to avoid clashing with the existing module
# of that name. The two may be resolved later.
import pandas as pd
import sys
sys.path.append("/opt/github.com/spond")

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl


from probabilistic_glove import ProbabilisticGloveLayer
from glove_layer import GloveLayer
from spond.models.mlp import MLP
from spond.metrics.mapping_accuracy import mapping_accuracy
from spond.experimental.openimages.readfile import readlabels

# hidden layer size, somewhat arbitrarily choose 100
HIDDEN = 100


def torch_mapping_misclassified_1nn(mapped, target, indexes, device='cpu'): #f_x, y):
    # mapped: mapped original embeddings, approximations of the target
    # target: actual target embedding
    # indexes: indexes describing
    #          which embeddings should be checked for correspondence
    #          meaning if indexes 2, 3, 6 are passed, we should check
    #          if the nearest neighbours of mapped[2, 3, 6] are
    #          target[2, 3, 6].

    # equivalent of mapping_accuracy but implemented in torch,
    # and only for 1nn. Returns the misclassification rather than accuracy
    # because we want to pass this to the optimiser for minimisation.
    # again, weird compute mode is needed because without it,
    # distance from an item to itself is not 0.
    dist = torch.cdist(mapped[indexes], target, compute_mode="donot_use_mm_for_euclid_dist").to(device)
    values, nn_inds = dist.topk(1, dim=1, largest=False)
    # we need to do something sneaky to count the mismatches.
    # If we just sum the mismatches, there is no backpropagation.
    # Therefore, we have to do something to force the f_x and y to be
    # involved in the calculation
    nn_inds = nn_inds.squeeze(dim=1)
    mismatches = nn_inds != torch.tensor(indexes).to(device)
    included = values.squeeze(dim=1)[mismatches]
    # trying a lot of things to basically end up with the number of items
    # that are mismatched.
    count = torch.ceil(included) - torch.floor(included)
    loss = count.sum() / len(indexes)
    return loss


class AlignedGloveLayer(nn.Module):

    def __init__(self,
                 x_cooc,                # co-occurrence matrix for x
                 x_embedding_dim,       # dimension of x
                 y_cooc,                # co-occurrence matrix for y
                 y_embedding_dim,       # dimension of y
                 index_map,              # list of pairs that map a concept in x
                                         # to a concept in y
                 seed=None,
                 probabilistic=False     # If set, use ProbabilisticGloveLayer
                 ):
        super(AlignedGloveLayer, self).__init__()
        self.seed = seed

        self.probabilistic = probabilistic
        x_nconcepts = x_cooc.size()[0]
        y_nconcepts = y_cooc.size()[0]
        kws = {}
        if seed is not None:
            if probabilistic:
                kws['seed'] = seed
            torch.manual_seed(seed)

        cls = ProbabilisticGloveLayer if probabilistic else GloveLayer
        self.x_emb = cls(x_nconcepts, x_embedding_dim, x_cooc, **kws)
        self.y_emb = cls(y_nconcepts, y_embedding_dim, y_cooc, **kws)
        self.index_map = index_map
        # This is stored just for speed
        #self.rev_index_map = {v: k for k, v in self.index_map.items()}

        # build the MLPs that are the aligner layers
        # f(x) --> y
        self.fx = MLP(x_embedding_dim, HIDDEN, output_size=y_embedding_dim)
        # g(y) --> x
        self.gy = MLP(y_embedding_dim, HIDDEN, output_size=x_embedding_dim)
        self.device = None

    def _init_samples(self):
        # TODO: not sure how this will work with different dataset sizes
        # for x and y.
        if self.probabilistic:
            self.x_emb._init_samples()
            self.y_emb._init_samples()

    def _set_device(self, device):
        self.device = device
        self.x_emb._set_device(device)
        self.y_emb._set_device(device)
        self.fx = self.fx.to(device)
        self.gy = self.gy.to(device)
        # TODO Not sure about the other items.

    def loss(self, x_inds, y_inds):
        # x_inds and y_inds are sequences of the x and y indices that form
        # this minibatch.
        # The loss contains the following items:
        # For all concepts:
        # 1. Glove loss for both x and y embeddings
        # recall that ProbabilisticGloveLayer will return a list of loss
        # There is no MSE loss here like there is in GloveLayer and
        # ProbabilisticGlove, because this is now no longer supervised.
        # We are not trying to train to match deterministic embeddings.
        self.losses = []
        self.losses += self.x_emb.loss(x_inds)
        self.losses += self.y_emb.loss(y_inds)
        x_mapped = self.fx(self.x_emb.weight)
        y_mapped = self.gy(self.y_emb.weight)
        # calculate cycle loss: g(f(x)) - x
        x_rt = self.gy(x_mapped)
        fx_diff = x_rt - self.x_emb.weight
        # This is the cycle loss: |f(g(x)) - x|
        # sum or mean? I have a feeling if we use sum, then this loss will
        # override the mismatch loss
        cycle_fx_loss = torch.sqrt(torch.einsum('ij,ij->i', fx_diff, fx_diff)).mean() #sum()
        self.losses.append(cycle_fx_loss)

        # other cycle loss: f(g(y)) - y
        y_rt = self.fx(y_mapped)
        gy_diff = y_rt - self.y_emb.weight
        cycle_gy_loss = torch.sqrt(torch.einsum('ij,ij->i', gy_diff, gy_diff)).mean() #sum()
        self.losses.append(cycle_gy_loss)
        # For concepts that exist in both domains:
        # The intersection will always be trained for mismatch / supervised loss.
        # and self.rev_index_map
        # This code relies heavily on the fact that the index_map is small,
        # therefore using numpy operations is fast.
        x_intersect = self.index_map[:, 0]
        y_intersect = self.index_map[:, 1]
        sup_diff_x = x_mapped[x_intersect] - self.y_emb.weight[y_intersect]
        sup_loss_x = torch.sqrt(torch.einsum('ij,ij->i', sup_diff_x, sup_diff_x)).mean()

        self.losses.append(sup_loss_x)
        # # 4. 1 if nearest neighbour of f(x) is not the known y mapping,
        # #    0 otherwise
        fx_mismatch = torch_mapping_misclassified_1nn(
            x_mapped, self.y_emb.weight, x_intersect, self.device
        )
        self.losses.append(fx_mismatch)
        sup_diff_y = y_mapped[y_intersect] - self.x_emb.weight[x_intersect]
        sup_loss_y = torch.sqrt(torch.einsum('ij,ij->i', sup_diff_y, sup_diff_y)).mean()
        self.losses.append(sup_loss_y)
        # # 5. 1 if nearest neighbour of g(y) is not the known x mapping,
        # #    0 otherwise
        gy_mismatch = torch_mapping_misclassified_1nn(
            y_mapped, self.x_emb.weight, y_intersect, self.device
        )
        self.losses.append(gy_mismatch)
        print(f"losses: {self.losses}")
        # Losses must be summed like below, or learning doesn't happen
        # cannot convert to a tensor containing self.losses,
        # even if requires_grad is set, learning does not happen
        # something wrong with the backpropagation.
        loss = sum(self.losses)
        return loss


class AlignedGlove(pl.LightningModule):

    # I decided to use the labels files as inputs because then
    # all the code for taking indices can be done inside this class.
    # For this to work, the indices of the entries in the *_cooc_file must
    # match the order of the labels.
    # That is, if x_cooc_file corresponds to the openimages file,
    # the first row of the cooccurrence matrix from this file
    # must correspond to the first label/name in x_labels_file.
    # Strictly speaking, merged_labels_file is not necessary as we could
    # merge the files internally too. It will be used to generate the
    # indices of corresponding concepts (eg. index of "Bird" in openimages and
    # index of "Bird" in audioset).
    def __init__(self,
                 batch_size,
                 data,  # DataDict class
                 # all files must be full paths
                 x_embedding_dim,  # dimension of x
                 y_embedding_dim,  # dimension of y
                 probabilistic=False,  # whether to use probabilistic layers
                 seed=None,
                 ):
        super(AlignedGlove, self).__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.data = data
        self.x_embedding_dim = x_embedding_dim
        self.y_embedding_dim = y_embedding_dim
        self.probabilistic = probabilistic

        self.aligner = AlignedGloveLayer(
            self.data.x_cooc,
            self.x_embedding_dim,
            self.data.y_cooc,
            self.y_embedding_dim,
            self.data.index_map,
            seed=seed,
            probabilistic=probabilistic
        )

    def additional_state(self):
        # return dictionary of things that were passed to constructor
        # should contain everything necessary to replicate a model.
        # we don't save things like the actual training data and so on
        # obviously this means that when the model is loaded,
        # the appropriate training file must be present.
        state = dict(
            seed=self.seed,
            batch_size=self.batch_size,
            x_embedding_dim=self.x_embedding_dim,
            y_embedding_dim=self.y_embedding_dim,
            probabilistic=self.probabilistic
        )
        state.update(self.data.state_dict())
        return state

    def save(self, filename):
        state = self.state_dict()
        state.update(self.additional_state())
        torch.save(state, filename)

    @classmethod
    def load(cls, filename, device='cpu'):
        state = torch.load(filename, map_location=device)
        data_items = (
            'x_cooc', 'x_labels_file', 'y_cooc', 'y_labels_file',
            'all_labels_file'
        )
        additional_state = {}
        for item in data_items:
            additional_state[item] = state.pop(item)
        data = DataDictionary(**additional_state)
        additional_state = {'data': data}
        items = (
            'seed', 'x_embedding_dim', 'y_embedding_dim', 'batch_size',
            'probabilistic'
        )
        for item in items:
            additional_state[item] = state.pop(item)
        instance = cls(**additional_state)
        instance.load_state_dict(state)
        return instance

    def forward(self, indices):
        # indices are a tuple of x and y index
        x_ind, y_ind = indices
        losses = self.aligner.loss(x_ind, y_ind)
        return losses

    def training_step(self, batch, batch_idx):
        # if this isn't done explicitly it somehow never gets set automatically
        # by lightning
        # self.device is set by Lightning, but it doesn't pass down
        # to the lower-level layers
        if self.aligner.device is None:
            self.aligner._set_device(self.device)
        # init samples every batch and not just on batch_idx = 0
        self.aligner._init_samples()

        # indices: the indices of the items present in this batch
        #          essentially meaningless because x_ind and y_ind are
        #          more important
        # x_ind: indices of x embeddings to be used in this batch
        # y_ind: indices of y embeddings to be used in this batch
        indices, xy_indices = batch
        x_ind = xy_indices[:, 0]
        y_ind = xy_indices[:, 1]
        # forward step
        losses = self.forward((x_ind, y_ind))
        print(f"losses: {losses}")
        loss = torch.sum(losses)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=0.005)
        return opt

    def train_dataloader(self):
        dataset = GloveDualDataset(self.data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class GloveDualDataset(Dataset):

    def __init__(self, data):
        self.data = data
        # Internally, Lightning will make index samples based on whatever
        # is returned by self.__len__().
        # This means that if len(self) = 2000 and batch size is 300,
        # Lightning will do whatever is necessary to make N batches of
        # 300 random indexes from 0 to 1999.
        #
        # We need to make sure all concepts are trained regardless of whether
        # they appear in audio, images, or both.
        #
        # There are many more openimages items than audioset
        # If we just sample from the total set of indices, we'll end up
        # with very imbalanced training.
        # There are 526 audio concepts and 19000+ openimage ones.
        #
        # Suppose we have batch size 100
        # In one epoch we have to go through ~1900 openimage batches and
        # 6 audio batches
        #
        # I think the right thing to do is to "unroll" the audio dataset
        # to make the total the same length as the openimage dataset.
        # Then, randomise the unrolled audio indexes so that
        # the pairing of openimage index with audio index is nondeterministic
        #
        # Then, one batch will be 100 pairs of indices from both sets,
        # and we should go randomly through multiple iterations of the
        # smaller dataset.
        x_n = self.data.x_n
        y_n = self.data.y_n
        self.x_n = x_n
        self.y_n = y_n
        self.N = max(x_n, y_n)
        larger = x_n if x_n > y_n else y_n
        smaller = y_n if x_n == larger else x_n
        self.N = larger
        times = math.ceil(larger/smaller)

        smaller_inds = np.repeat([np.arange(smaller)], times, axis=0).ravel()[:larger]
        np.random.shuffle(smaller_inds)
        self.inds = torch.arange(self.N)
        # TODO: fix this, there's an assumption that x is larger than y
        self.out = torch.vstack([torch.arange(larger), torch.tensor(smaller_inds)]).T

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # This must return whatever we map idx to internally.
        # It doesn't matter what is returned as long as we
        # unpack it correctly inside the training loop.
        inds = self.inds[idx]
        out = self.out[idx]
        return inds, out


class DataDictionary:
    # This class takes the co-occurrence matrices and labels files
    # and does some sanity checks, as well as storing various useful things.

    def __init__(self,
                 x_cooc,      # co-occurrence matrix for x
                 x_labels_file,    # full filepath or handle containing all x labels to names
                 y_cooc,      # co-occurrence matrix for y
                 y_labels_file,    # full filepath or handle containing all y labels to names
                 all_labels_file,        # full filepath or handle containing all labels to names
                 ):

        x_nconcepts = x_cooc.size()[0]
        y_nconcepts = y_cooc.size()[0]

        self.x_labels_file = x_labels_file
        self.y_labels_file = y_labels_file
        self.x_cooc = x_cooc
        self.y_cooc = y_cooc

        self.x_n = x_nconcepts
        self.y_n = y_nconcepts
        self.all_labels_file = all_labels_file
        all_labels, all_names = readlabels(all_labels_file, rootdir=None)
        name_to_label = {v: k for k, v in all_names.items()}
        index_to_label = {v: k for k, v in all_labels.items()}
        index_to_name = {v: all_names[k] for k, v in all_labels.items()}
        name_to_index = {v: k for k, v in index_to_name.items()}

        # x_labels is dictionary of label to index
        # x_names is dictionary of label to name
        x_labels, x_names = readlabels(x_labels_file, rootdir=None)
        assert len(x_labels) == x_nconcepts, (
            f"x co-occurrence does not contain the same number of concepts as the labels file: "
            f"\nExpected {x_nconcepts} but got {len(x_labels)}"
        )
        # same applies to y_labels and y_names
        y_labels, y_names = readlabels(y_labels_file, rootdir=None)
        assert len(y_labels) == y_nconcepts, (
            f"y co-occurrence does not contain the same number of concepts as the labels file: "
            f"\nExpected {y_nconcepts} but got {len(y_labels)}"
        )

        y_name_to_label = {v: k for k, v in y_names.items()}
        intersection = {}
        for x_label, x_name in x_names.items():
            if x_label in y_names:
                # we have to use labels as the intersection,
                # because there are multiple labels with the same name.
                # for example /m/07qcpgn is Tap in audioset, meaning the sound Tap
                # but /m/02jz0l is Tap in openimages meaning the object Tap.
                intersection[x_label] = (x_labels[x_label], y_labels[y_name_to_label[x_name]])
        # Tuple of (index in all, index in x, index in y)
        # TODO: ugh, too many levels of indirection, clean up later.
        index_map = list(intersection.values())
        x_indexes = torch.tensor([all_labels[label] for label in x_labels])
        y_indexes = torch.tensor([all_labels[label] for label in y_labels])

        self.intersection_names = intersection

        # universe: keys = labels, values = index into universe
        self.all_labels = all_labels
        # universe: keys = labels, values = names
        self.all_names = all_names

        # indexes of x concepts in the universe
        self.x_indexes = x_indexes
        # indexes of y concepts in the universe
        self.y_indexes = y_indexes

        # given an index in the universe, stores the mapping from
        # index into universe to index in x and y files
        # eg.  if universe index 5 is Cat and x index 0 is Cat and
        # y index 77 is Cat then
        # self.union_indexes contains
        #  (5, 0, 77)
        self.intersection_indexes = torch.tensor([
            (
                all_labels[label],
                x_labels[label],
                y_labels[label]
            )
            for label in intersection
        ])

        self.index_map = self.intersection_indexes[:, 1:].numpy()

    def state_dict(self):
        # this will only work if all the files are strings
        check_files = (self.x_labels_file, self.y_labels_file, self.all_labels_file)
        if not all([isinstance(f, str) for f in check_files]):
            raise AssertionError("Can only persist this item if all filenames are strings")

        return dict(
            x_cooc=self.x_cooc,      # co-occurrence matrix for x
            x_labels_file=self.x_labels_file,    # full filepath or handle containing all x labels to names
            y_cooc=self.y_cooc,      # co-occurrence matrix for y
            y_labels_file=self.y_labels_file,    # full filepath or handle containing all y labels to names
            all_labels_file=self.all_labels_file        # full filepath or handle containing all labels to names
        )


if __name__ == '__main__':

    import os
    import socket
    remote = socket.gethostname().endswith('pals.ucl.ac.uk')
    if remote:
        # set up pythonpath
        ppath = '/home/petra/spond'
        # set up data pth
        datapath = '/home/petra/data'
        gpu = True
    else:
        ppath = '/opt/github.com/spond/spond/experimental'
        datapath = ppath
        gpu = False

    sys.path.append(ppath)

    seed = 1
    trainer = pl.Trainer(gpus=int(gpu), max_epochs=500, progress_bar_refresh_rate=20)
    # batch sizes larger than 100 causes a strange CUDA error with pytorch 1.7
    # Had to upgrade to pytorch 1.9
    # It may be due to some internal array being larger than 65535 when cdist is used.
    # https://github.com/pytorch/pytorch/issues/49928
    # https://discuss.pytorch.org/t/cuda-invalid-configuration-error-on-gpu-only/50399/15
    batch_size = 1000
    y_cooc_file = os.path.join(datapath, 'audioset', "co_occurrence_audio_all.pt")
    y_labels_file = os.path.join(datapath, 'audioset', "class_labels.csv")
    y_dim = 6
    if remote:
        x_cooc_file = os.path.join(datapath, 'openimages', "co_occurrence.pt")
        x_labels_file = os.path.join(datapath, 'openimages', "oidv6-class-descriptions.csv")
        x_dim = 30
        all_labels_file = os.path.join(datapath, "all_labels.csv")
    else:
        # train audioset against itself
        x_cooc_file = y_cooc_file
        x_labels_file = y_labels_file
        x_dim = y_dim
        all_labels_file = x_labels_file

    datadict = DataDictionary(
        x_cooc=torch.load(x_cooc_file),
        x_labels_file=x_labels_file,
        y_cooc=torch.load(y_cooc_file),
        y_labels_file=y_labels_file,
        all_labels_file=all_labels_file
    )
    # temporarily: hack the index map so only a few concepts are aligned
    #datadict.index_map = datadict.index_map[:5]
    model = AlignedGlove(batch_size,
                         data=datadict,
                         x_embedding_dim=x_dim,  # dimension of x
                         y_embedding_dim=y_dim,  # dimension of y
                         seed=seed,
                         probabilistic=False)
    trainer.fit(model)
    #model.save('aligned.pt')
    #model_rt = AlignedGlove.load('aligned.pt')
