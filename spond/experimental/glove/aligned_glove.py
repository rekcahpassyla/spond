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


def torch_mapping_misclassified_1nn(f_x, y):
    # equivalent of mapping_accuracy but implemented in torch,
    # and only for 1nn. Returns the misclassification rather than accuracy
    # because we want to pass this to the optimiser for minimisation.
    # again, weird compute mode is needed because without it,
    # distance from an item to itself is not 0.
    dist = torch.cdist(f_x, y, compute_mode="donot_use_mm_for_euclid_dist")
    values, inds = dist.sort(dim=0)
    # None of the following backpropagate in torch
    # probably because there are no gradients defined
    # we need to work out how to define something that has a gradient
    #nn_indices = inds[0]
    #nconcepts = f_x.size()[0]
    #mismatches = torch.arange(nconcepts) != nn_indices
    #loss = mismatches.float().mean()
    #return loss
    loss = values[0].mean()
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
        cls = ProbabilisticGloveLayer if probabilistic else GloveLayer
        self.x_emb = cls(x_nconcepts, x_embedding_dim, x_cooc)
        self.y_emb = cls(y_nconcepts, y_embedding_dim, y_cooc)
        # dictionary of x to y
        self.index_map = dict(index_map)
        # This is stored just for speed
        self.rev_index_map = {v: k for k, v in self.index_map.items()}

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
        # For concepts that exist in both domains:
        # First we have to find the concepts in this index
        # that exist in both domains, by comparing with self.index_map
        # and self.rev_index_map
        # This code relies heavily on the fact that the index_map and
        # rev_index_map are small, and therefore using numpy operations
        # is fast.

        x_present = list(set(self.index_map.keys()).intersection(set(x_inds.tolist())))
        # we also need the y values that x_present mapped to
        y_check = [self.index_map[k] for k in x_present]
        y_present = list(set(self.rev_index_map.keys()).intersection(
            set(y_inds.tolist())))
        x_check = [self.rev_index_map[k] for k in y_present]
        # 2. distance between fx and y_embedding
        fx_input = self.x_emb.weight[x_present]
        fx_output = self.fx(fx_input)
        x_rt = self.gy(fx_output)
        fx_diff = x_rt - fx_input
        fx_loss = torch.sqrt(torch.einsum('ij,ij->i', fx_diff, fx_diff))
        # must be mean because the batch size may differ
        self.losses.append(fx_loss.mean())
        # 3. distance between gy and x_embedding
        gy_input = self.y_emb.weight[y_present]
        gy_output = self.gy(gy_input)
        y_rt = self.fx(gy_output)
        gy_diff = y_rt - gy_input
        gy_loss = torch.sqrt(torch.einsum('ij,ij->i', gy_diff, gy_diff))
        #gy_expected_output = self.x_emb.weight[x_check]
        #gy_diff = gy_output - gy_expected_output
        #gy_loss = torch.sqrt(torch.einsum('ij,ij->i', gy_diff, gy_diff))
        self.losses.append(gy_loss.mean())
        # TODO: add these other losses later.
        # 4. 1 if nearest neighbour of f(x) is not the known y mapping,
        #    0 otherwise
        # TODO: try to convert to torch, because this will be slow
        fx_mismatch = torch_mapping_misclassified_1nn(
            fx_output,
            self.y_emb.weight[y_check]
        )
        # 5. 1 if nearest neighbour of g(y) is not the known x mapping,
        #    0 otherwise
        gy_mismatch = torch_mapping_misclassified_1nn(
            gy_output,
            self.x_emb.weight[x_check]
        )
        self.losses.append(fx_mismatch)
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
                 # all files must be full paths
                 x_cooc_file,      # co-occurrence file for x
                 x_embedding_dim,  # dimension of x
                 x_labels_file,    # file containing all x labels to names
                 y_cooc_file,      # co-occurrence file for y
                 y_embedding_dim,  # dimension of y
                 y_labels_file,    # file containg all y labels to names
                 all_labels_file,        # file containing all labels to names
                 seed=None,
                 ):
        super(AlignedGlove, self).__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.x_cooc_file = x_cooc_file
        self.x_cooc = torch.load(x_cooc_file)
        self.x_embedding_dim = x_embedding_dim
        self.x_labels_file = x_labels_file
        self.y_cooc_file = y_cooc_file
        self.y_cooc = torch.load(y_cooc_file)
        self.y_embedding_dim = y_embedding_dim
        self.y_labels_file = y_labels_file
        self.all_labels_file = all_labels_file

        # labels = machine IDs to index
        # names = machine IDs to names
        all_labels, all_names = readlabels(all_labels_file, rootdir=None)

        name_to_label = {v: k for k, v in all_names.items()}
        index_to_label = {v: k for k, v in all_labels.items()}
        index_to_name = {v: all_names[k] for k, v in all_labels.items()}
        name_to_index = {v: k for k, v in index_to_name.items()}

        # x_labels is dictionary of label to index
        # x_names is dictionary of label to name
        x_labels, x_names = readlabels(self.x_labels_file, rootdir=None)
        # same applies to y_labels and y_names
        y_labels, y_names = readlabels(self.y_labels_file, rootdir=None)
        y_name_to_label = {v: k for k, v in y_names.items()}
        # the number of labels/names corresponds to the dimension of the
        # co-occurrence matrix, and represents the number of concepts.
        # Now we have to find the indices of the union of concepts
        # in the x and y domains.
        # First find the union
        # TODO: these currently contain redundant information,
        # but clean them up later when we know better how they are used
        union = {}
        for x_label, x_name in x_names.items():
            if x_label in y_names:
                union[x_name] = (x_labels[x_label], y_labels[y_name_to_label[x_name]])
        self.union_names = union
        # Tuple of (index in all, index in x, index in y)
        # TODO: ugh, too many levels of indirection, clean up later.
        self.union_indexes = torch.tensor([
            (
                name_to_index[name],
                x_labels[name_to_label[name]],
                y_labels[name_to_label[name]],
            )
            for name in union
        ])

        index_map = list(self.union_names.values())

        self.aligner = AlignedGloveLayer(
            self.x_cooc,
            self.x_embedding_dim,
            self.y_cooc,
            self.y_embedding_dim,
            index_map,
            seed=seed
        )

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
        # batch_idx is the batch number in this epoch,
        # runs from 0 to datasize/batchsize
        if batch_idx == 0:
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
        #loss = torch.sum(losses)
        #print(f"loss: {loss}")
        #return loss
        print(f"losses: {losses}")
        loss = torch.sum(losses)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=0.005)
        #opt = optim.Adagrad(self.parameters(), lr=0.05)
        return opt

    def train_dataloader(self):
        dataset = GloveDualDataset(self.x_cooc.size()[0], self.y_cooc.size()[0])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class GloveDualDataset(Dataset):

    def __init__(self, x_n, y_n):
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
        self.x_n = x_n
        self.y_n = y_n
        self.N = max(x_n, y_n)
        larger = x_n if x_n > y_n else y_n
        smaller = y_n if x_n == larger else x_n
        self.N = larger
        times = math.ceil(larger/smaller)

        smaller_inds = np.repeat([np.arange(smaller)], times, axis=0).ravel()[:larger]
        np.random.shuffle(smaller_inds)
        self.x = torch.arange(self.N)
        self.y = torch.vstack([torch.arange(larger), torch.tensor(smaller_inds)]).T

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # This must return whatever we map idx to internally.
        # It doesn't matter what is returned as long as we
        # unpack it correctly inside the training loop.
        x = self.x[idx]
        y = self.y[idx]
        return x, y


if __name__ == '__main__':
    seed = 1
    trainer = pl.Trainer(gpus=0, max_epochs=2000, progress_bar_refresh_rate=20)
    batch_size = 100
    # train audioset against itself
    cooc_file = "/opt/github.com/spond/spond/experimental/audioset/co_occurrence_audio_all.pt"
    labels_file = "/opt/github.com/spond/spond/experimental/audioset/class_labels.csv"
    dim = 6
    model = AlignedGlove(batch_size,
                         # all files must be full paths
                         x_cooc_file=cooc_file,      # co-occurrence file for x
                         x_embedding_dim=dim,  # dimension of x
                         x_labels_file=labels_file,    # file containing all x labels to names
                         y_cooc_file=cooc_file,      # co-occurrence file for y
                         y_embedding_dim=dim,  # dimension of y
                         y_labels_file=labels_file,    # file containg all y labels to names
                         all_labels_file=labels_file,        # file containing all labels to names
                         seed=seed)
    trainer.fit(model)

