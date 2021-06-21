# probabilistic embeddings
import itertools
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F


import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import socket
if socket.gethostname().endswith('pals.ucl.ac.uk'):

    import sys
    sys.path.append('/home/petra/spond')

from spond.experimental.glove.glove_layer import GloveEmbeddingsDataset


class ProbabilisticGloveLayer(nn.Embedding):

    # TODO: Is there a way to express constraints on weights other than
    # the nn.Functional.gradient_clipping ?
    # ANSWER: Yes, if you use Pyro with constraints.
    def __init__(self, num_embeddings, embedding_dim, co_occurrence,
                 # glove learning options
                 x_max=100, alpha=0.75,
                 seed=None, # if None means don't set
                 # whether or not to use the wi and wj
                 # if set to False, will use only wi
                 double=False,
                 # nn.Embedding options go here
                 padding_idx=None,
                 scale_grad_by_freq=None,
                 max_norm=None, norm_type=2,
                 sparse=False   # not supported- just here to keep interface
                 ):
        self.seed = seed
        if seed is not None:
            # internal import to allow setting seed before any other imports
            import pyro
            pyro.set_rng_seed(seed)
        # Internal import because we need to set seed first
        from pyro.distributions import MultivariateNormal
        # Not calling Embedding constructor, as this module is
        # composed of Embeddings. However we have to set some attributes
        nn.Embedding.__init__(self, num_embeddings, embedding_dim,
                              padding_idx=None, max_norm=None,
                              norm_type=2.0, scale_grad_by_freq=False,
                              sparse=False, _weight=None)
        if sparse:
            raise NotImplementedError("`sparse` is not implemented for this class")
        # for the total weight to have a max norm of K, the embeddings
        # that are summed to make them up need a max norm of K/2
        used_norm = max_norm / 2 if max_norm else None
        kws = {}
        if used_norm:
            kws['max_norm'] = used_norm
            kws['norm_type'] = norm_type
        if padding_idx:
            kws['padding_idx'] = padding_idx
        if scale_grad_by_freq is not None:
            kws['scale_grad_by_freq'] = scale_grad_by_freq
        # double is not supported, but we keep the same API.
        assert not double, "Probabilistic embedding can only be used in single mode"
        # This assumes each dimension is independent of the others,
        # and that all the embeddings are independent of each other.
        # We express it as MV normal because this allows us to use a
        # non diagonal covariance matrix
        # try setting the variance low here instead

        self.wi_dist = MultivariateNormal(
            torch.zeros((num_embeddings, embedding_dim)),
            # changing this to 1e-9 makes the embeddings converge to something
            # else than the pretrained
            torch.eye(embedding_dim)# * 1e-9
        )
        self.bi_dist = MultivariateNormal(
            torch.zeros((num_embeddings, 1)),
            torch.eye(1)
        )
        # Deterministic means for the weights and bias, that will be learnt
        # means will be used to transform the samples from the above wi/bi
        # samples.
        # Express them as embeddings because that sets up all the gradients
        # for backprop, and allows for easy indexing.
        self.wi_mu = nn.Embedding(num_embeddings, embedding_dim)
        self.wi_mu.weight.data.uniform_(-1, 1)
        self.bi_mu = nn.Embedding(num_embeddings, 1)
        self.bi_mu.weight.data.zero_()
        # wi_sigma = log(1 + exp(wi_rho)) to enforce positivity.
        self.wi_rho = nn.Embedding(num_embeddings, embedding_dim)
        # initialise the rho so softplus results in small values
        # 1e-9 - this appears to be about -20 so we have to re-center around -19?
        # except it doesn't work- re-centering just makes the means nowhere near
        self.wi_rho.weight.data.uniform_(-1, 1) #- 19
        self.bi_rho = nn.Embedding(num_embeddings, 1)
        self.bi_rho.weight.data.zero_()
        # using torch functions should ensure backprop is set up right
        self.softplus = nn.Softplus()
        #self.wi_sigma = softplus(self.wi_rho.weight) #torch.log(1 + torch.exp(self.wi_rho))
        #self.bi_sigma = softplus(self.bi_rho.weight) #torch.log(1 + torch.exp(self.bi_rho))

        self.co_occurrence = co_occurrence.coalesce()
        # it is not very big
        self.coo_dense = self.co_occurrence.to_dense()
        self.x_max = x_max
        self.alpha = alpha
        # Placeholder. In future, we will make an abstract base class
        # which will have the below attribute so that all instances
        # carry their own loss.
        self.losses = []
        self._setup_indices()
        # This can only be set up later after the trainer has initialised
        self.device = None

    def _setup_indices(self):
        # Do some preprocessing to make looking up indices faster.
        # The co-occurrence matrix is a large array of pairs of indices
        # In the course of training, we will be given a list of
        # indices, and we need to find the pairs that are present.
        self.allindices = self.co_occurrence.indices()
        N = self.allindices.max() + 1
        # Store a dense array of which pairs are active
        # It is booleans so should be small even if there are a lot of tokens
        self.allpairs = torch.zeros((N, N), dtype=bool)
        self.allpairs[(self.allindices[0], self.allindices[1])] = True
        self.N = N

    def _set_device(self, device):
        self.device = device
        self.softplus = self.softplus.to(device)
        self.wi_mu = self.wi_mu.to(device)
        self.bi_mu = self.bi_mu.to(device)
        self.wi_rho = self.wi_rho.to(device)
        self.bi_rho = self.bi_rho.to(device)
        self.co_occurrence = self.co_occurrence.to(device)
        self.coo_dense = self.coo_dense.to(device)
        self.allpairs = self.allpairs.to(device)

    def weights(self, n=1):
        # we are taking one sample from each embedding distribution
        if n == 1:
            sample_shape = torch.Size([])
        else:
            sample_shape = torch.Size([n])
        wi_eps = self.wi_dist.sample(sample_shape).to(self.device)
        # TODO: Only because we have assumed a diagonal covariance matrix,
        # is the below elementwise multiplication (* rather than @).
        # If it was not diagonal, we would have to do matrix multiplication
        #wi = self.wi_mu + wi_eps * self.wi_sigma
        wi = (
            self.wi_mu.weight +
            # multiplying by 1e-9 below should have the same effect
            # as changing the wi_eps variance to 1e-9, but it doesn't.
            # multiplying here results in wi_mu converging very closely
            # to the deterministic embeddings, but the wi_sigma variance remains
            # the same as in the other case.
            wi_eps * self.softplus(self.wi_rho.weight) #* 1e-9
        )
        return wi

    # implemented as such to be consistent with nn.Embeddings interface
    def forward(self, indices):
        return self.weights()(indices)

    def _update(self, i_indices, j_indices):
        # we need to do all the sampling here.
        # TODO: Not sure what to do with j_indices. Do we update the j_indices
        # TODO: Only because we have assumed a diagonal covariance matrix,
        # is the below elementwise multiplication (* rather than @).
        # If it was not diagonal, we would have to do matrix multiplication
        w_i = (
            self.wi_mu(i_indices) +
            self.wi_eps[i_indices] * self.softplus(self.wi_rho(i_indices))
        )

        b_i = (
            self.bi_mu(i_indices) +

            self.bi_eps[i_indices] * self.softplus(self.bi_rho(i_indices))
        ).squeeze()
        # If the double updating is not done, it takes a long time to converge.
        w_j = (
                self.wi_mu(j_indices) +
                self.wi_eps[j_indices] * self.softplus(self.wi_rho(j_indices))
        )
        b_j = (
                self.bi_mu(j_indices) +
                self.bi_eps[j_indices] * self.softplus(self.bi_rho(j_indices))
        ).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        #else:
        #    x = torch.sum(w_i, dim=1) + b_i
        #    #x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        return x

    def _init_samples(self):
        # On every 0th batch in an epoch, sample everything.
        sample_shape = torch.Size([])
        self.wi_eps = self.wi_dist.sample(sample_shape) #* 1e-9
        self.bi_eps = self.bi_dist.sample(sample_shape) #* 1e-9
        self.wi_eps = self.wi_eps.to(self.device)
        self.bi_eps = self.bi_eps.to(self.device)


    def _loss_weights(self, x):
        # x: co_occurrence values
        wx = (x/self.x_max)**self.alpha
        wx = torch.min(wx, torch.ones_like(wx))
        return wx

    def loss(self, indices):
        # inputs are indexes, targets are actual embeddings
        # In the actual algorithm, "inputs" is the weight_func run on the
        # co-occurrence statistics.
        # loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
        # not sure what it should be replaced by here
        # "targets" are the log of the co-occurrence statistics.
        # need to make every pair of indices that exist in co-occurrence file
        # Not every index will be represented in the co_occurrence matrix
        # To calculate glove loss, we will take all the pairs in the co-occurrence
        # that contain anything in the current set of indices.
        # There is a disconnect between the indices that are passed in here,
        # and the indices of all pairs in the co-occurrence matrix
        # containing those indices.
        indices = indices
        indices = indices.sort()[0]
        subset = self.allpairs[indices]
        if not torch.any(subset):
            self.losses = [0]
            return self.losses
        # now look up the indices of the existing pairs
        # it is faster to do the indexing into an array of bools
        # instead of the dense array
        subset_indices = torch.nonzero(subset)
        i_indices = indices[subset_indices[:, 0]]
        j_indices = subset_indices[:, 1]

        i_indices = i_indices
        j_indices = j_indices

        targets = self.coo_dense[(i_indices, j_indices)]
        weights_x = self._loss_weights(targets)
        current_weights = self._update(i_indices, j_indices)
        loss = weights_x * F.mse_loss(
            current_weights, torch.log(targets), reduction='none')
        # This is a feasible strategy for mapping indices -> pairs
        # Second strategy: Loop over all possible pairs
        # More computationally intensive
        # Allow this to be configurable?
        # Degrees of separation but only allow 1 or -1
        # -1 is use all indices
        # 1 is use indices
        # We may want to save this loss as an attribute on the layer object
        # Does Lightning have a way of representing layer specific losses
        # Define an interface by which we can return loss objects
        # probably stick with self.losses = [loss]
        # - a list - because then we can do +=
        loss = torch.mean(loss)
        self.losses = [loss]
        return self.losses

    def entropy(self):
        # Calculate entropy based on the learnt rho (which must be transformed
        # to a variance using softplus)
        # entropy of a MV Gaussian =
        # 0.5 * N * ln (2 * pi * e) + 0.5 * ln (det C)
        N = self.embedding_dim
        C = self.softplus(self.wi_rho.weight)
        # diagonal covariance so just multiply all the items to get determinant
        # convert to log space so we can add across the dimensions
        # We don't need to convert back to exp because we need log det C anyway
        logdetC = torch.log(C).sum(axis=1)
        entropy = 0.5*(N*np.log(2*np.pi * np.e) + logdetC)
        return entropy

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # eventually, load this from a .pt file that contains all the
        # wi/wj/etc.
        raise NotImplementedError('Not yet implemented')


class ProbabilisticGlove(pl.LightningModule):

    def __init__(self, train_embeddings_file, batch_size, train_cooccurrence_file,
                 seed=None,
                 limit=None):
        # train_embeddings_file: the filename contaning the pre-trained weights
        # train_cooccurrence_file: the filename containing the co-occurrence statistics
        # that we want to match these embeddings to.
        super(ProbabilisticGlove, self).__init__()
        self.seed = seed
        self.limit = limit
        self.train_embeddings_file = train_embeddings_file
        self.train_cooccurrence_file = train_cooccurrence_file
        self.train_data = torch.load(train_embeddings_file)
        self.train_cooccurrence = torch.load(train_cooccurrence_file)
        self.batch_size = batch_size
        nemb, dim = self.train_data['wi.weight'].shape
        self.pretrained_embeddings = (
                self.train_data['wi.weight'] +
                self.train_data['wj.weight']
        )
        if limit:
            nemb = limit
        self.num_embeddings = nemb
        self.embedding_dim = dim
        # Need to call _set_device later.
        # We cannot create the GloveLayer later, because then some initialisation
        # doesn't happen and the optimiser will blow up.
        self.glove_layer = ProbabilisticGloveLayer(
            self.num_embeddings, self.embedding_dim,
            self.train_cooccurrence, seed=self.seed)

    def additional_state(self):
        # return dictionary of things that were passed to constructor
        # should contain everything necessary to replicate a model.
        # we don't save things like the actual training data and so on
        # obviously this means that when the model is loaded,
        # the appropriate training file must be present.
        state = dict(
            seed=self.seed,
            limit=self.limit,
            train_embeddings_file=self.train_embeddings_file,
            train_cooccurrence_file=self.train_cooccurrence_file,
            batch_size=self.batch_size,
        )
        return state

    def save(self, filename):
        # this is the torch model stuff
        state = self.state_dict()
        # this is the custom state like embeddings file name, etc
        state.update(self.additional_state())
        torch.save(state, f"{filename}")

    @classmethod
    def load(cls, filename, device='cpu'):
        state = torch.load(filename, map_location=device)
        # get the items that would have been passed to the constructor
        additional_state = {}
        items = (
            'seed', 'limit', 'train_embeddings_file',
            'train_cooccurrence_file', 'batch_size'
        )
        for item in items:
            additional_state[item] = state.pop(item)
        instance = cls(**additional_state)
        instance.load_state_dict(state)
        return instance

    def forward(self, indices):
        return self.glove_layer.weights()(indices)

    def training_step(self, batch, batch_idx):
        # if this isn't done explicitly it somehow never gets set automatically
        # by lightning
        if self.glove_layer.device is None:
            self.glove_layer._set_device(self.device)
        # If the batch_idx is 0, then we want to sample everything at once
        # if we do multiple samples, end up with torch complaining we are
        # trying to do backprop more than once.
        if batch_idx == 0:
            self.glove_layer._init_samples()

        # input: indices of embedding
        # targets: target embeddings
        indices, targets = batch
        # Ideal outcome:
        # Pytorch object that is substitutable with
        # nn.Embedding
        # Subclass of nn.Embedding
        # but internally, we should give the option to use
        # a co-occurrence matrix
        out = self.glove_layer.weights()[indices]
        loss = F.mse_loss(out, targets)
        glove_layer_loss = self.glove_layer.loss(indices)
        loss += glove_layer_loss[0]
        # How would this be used:
        # Take 2 domains with co-occurrence
        # Figure out intersection
        # Put these 2 layers into B-network
        # 2 branches which will connect at the top
        # Say we have L and R where L = openimages, R = something else
        # Top layer will combine audioset and openimage
        # and will be the "alignment layer"
        # Will take collective set of all concepts in both domains
        # The aligner layer will backpropagate down
        # to the respective embedding layers
        print(f"loss: {loss}")
        return loss

    def configure_optimizers(self):
        # Is there an equivalent configure_losses?
        # TODO: This is a torch optimiser. If we change to Pyro,
        # have to change to Pyro optimiser.
        # TODO: The learning rate probably needs to change,
        # because we are now sampling and it's going to be very stochastic.
        opt = optim.Adam(self.parameters(), lr=0.005)
        return opt

    def train_dataloader(self):
        dataset = GloveEmbeddingsDataset(self.train_data, self.limit)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class Similarity:
    # this class will take a root directory and a class and a list of ints (the seeds)
    # and it will load the models contained in that directory
    # and calculate the similarity matrices required.
    # The convention is that the models will be stored in a directory/file structure
    # as follows:
    # dirname/classname/classname_seed.pt
    # for example
    # dirname/ProbabilisticGlove/ProbabilisticGlove_2.pt
    # is the ProbabilisticGlove model run with seed set to 2.
    def __init__(self, dirname, clsobj, seedvalues):
        self.dirname = dirname
        self.clsobj = clsobj
        self.seedvalues = seedvalues
        self.models = {}

    def _load(self, seed):
        # load the model with the corresponding seed.
        # don't load all at once as this can lead to out of memory
        clsname = self.clsobj.__name__
        filename = os.path.join(dirname, clsname, f'{clsname}_{seed}.pt')
        model = self.clsobj.load(filename)
        return model

    def means(self, kernel, outfile, mode='a', mask=None):
        # kernel: callable that takes 2 arrays and returns similarity matrix
        # outfile: target file name
        # mask: if passed, should be a sequence of integers for which the
        # similarity will be calculated. It is up to the user to keep track
        # of what the final output indices mean.
        # Similarity will be calculated for each seed and stored in `outfile`
        store = pd.HDFStore(outfile, mode=mode)
        for seed in self.seedvalues:
            model = self._load(seed)
            values = model.glove_layer.wi_mu.weight.detach().numpy()
            if mask is not None:
                values = values[mask]
            thisvalue = kernel(values, values)
            store[str(seed)] = pd.DataFrame(thisvalue)
        store.close()

    def variances(self):
        # Load all the models, and calculate entropy

        # kernel: callable that takes 2 arrays and returns similarity matrix
        # Similarity will be calculated for each seed and stored in `outfile`

        # kernel: callable that takes 2 arrays and returns pairwise matrix
        # This function will sample nsamples occurrences of the weights
        # for each of 2 models, and calculate nsamples of the similarity matrix
        # then take the mean.
        store = pd.HDFStore(outfile)
        # internal import so we set the seed - does this work after other pyro imports?
        import pyro
        pyro.set_rng_seed(1)
        for i, seed1 in enumerate(self.seedvalues):
            for seed2 in self.seedvalues[i+1:]:
                model1 = self._load(seed1)
                model2 = self._load(seed2)
                total = None
                for i in range(nsamples):
                    thisvalue = kernel(
                        model1.glove_layer.weights().detach().numpy()[:100],
                        model2.glove_layer.weights().detach().numpy()[:100]
                    )
                    if total is None:
                        total = thisvalue
                    else:
                        total *= i-1
                        total += thisvalue
                        total /= i
                store[f"{seed1}x{seed2}"] = pd.DataFrame(total)
        store.close()

if __name__ == '__main__':
    import os
    import kernels
    if False:
        rdir = 'results/ProbabilisticGlove'
        model = ProbabilisticGlove.load(os.path.join(rdir, 'ProbabilisticGlove_1.pt'))
        samples = model.glove_layer.weights(n=5)
    if True:
        import sys
        ppath = '/opt/github.com/spond/spond/experimental'
        sys.path.append(ppath)

        from openimage.readfile import readlabels

        labelsfn = "/opt/github.com/spond/spond/experimental/audioset/all_labels.csv"

        labels, names = readlabels(labelsfn, rootdir=None)

        # now we need to find the indexes of the audio labels in the all-labels file
        audio_labels = pd.read_csv("/opt/github.com/spond/spond/experimental/audioset/class_labels_indices.csv",
                                   index_col=0)
        keep = np.array([labels[label] for label in audio_labels['mid'].values])

        dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        sim = Similarity(dirname, ProbabilisticGlove, seedvalues=(1, 2, 3, 4, 5))
        #sim.means(kernels.exponential, os.path.join(dirname, 'means_exponential.hdf5'))
        sim.means(kernels.dot, os.path.join(dirname, 'means_dot.hdf5'), mask=keep, mode='w')
    if False:
        seed = 1

        # change to gpus=1 to use GPU. Otherwise CPU will be used
        trainer = pl.Trainer(gpus=0, max_epochs=100, progress_bar_refresh_rate=20)
        # Trainer must be created before model, because we need to detect
        # what we requested for GPU.

        model = ProbabilisticGlove('glove_audio.pt', batch_size=100,
                                   seed=seed,
                                   train_cooccurrence_file='../audioset/co_occurrence_audio_all.pt')
        trainer.fit(model)
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        clsname = model.__class__.__name__
        outfile = os.path.join(outdir, clsname, f'{clsname}_{seed}.pt')
        model.save(outfile)

        #roundtrip = ProbabilisticGlove.load('/tmp/test.pt')



