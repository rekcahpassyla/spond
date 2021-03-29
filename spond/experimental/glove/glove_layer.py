# Test bed for Glove loss


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

#from utils import setup_torch

#device = setup_torch()


class GloveLayer(nn.Embedding):
    # At the moment, this does not extend nn.Embedding,
    # because there are many methods of nn.Embedding that do not make sense
    # for this, such as the ability to load from a pretrained set of
    # weights. Other arguments like max_norm, sparse, and so on,
    # also do not make sense for this.
    # Work out what subset of features make sense and implement those
    # Is there a way to express constraints on weights other than
    # the nn.Functional.gradient_clipping ?
    def __init__(self, num_embeddings, embedding_dim, co_occurrence,
                 x_max=100, alpha=0.75, device='cpu',
                 # nn.Embedding options go here
                 padding_idx=None,
                 scale_grad_by_freq=None,
                 max_norm=None, norm_type=2,
                 sparse=False   # not supported- just here to keep interface
                 ):
        # Not calling Embedding constructor, as this module is
        # composed of Embeddings. 
        nn.Module.__init__(self)
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
        self.wi = nn.Embedding(num_embeddings, embedding_dim, **kws)
        self.wj = nn.Embedding(num_embeddings, embedding_dim, **kws)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

        self.co_occurrence = co_occurrence.coalesce()
        self.device = device
        self.x_max = x_max
        self.alpha = alpha

    @property
    def weights(self):
        return self.wi.weight + self.wj.weight

    # implemented as such to be consistent with nn.Embeddings interface
    def forward(self, indices):
        return self.weights[indices]

    def _update(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x

    def _loss_weights(self, x):
        # x: co_occurrence values
        wx = (x/self.x_max)**self.alpha
        wx = torch.min(wx, torch.ones_like(wx))
        return wx.to(self.device)

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
        allindices = self.co_occurrence.indices()
        # The following is terrible, but figure it out once we get it working
        # Converting to numpy is very slow
        intersection = set(indices.numpy()).intersection(
            set(allindices.unique().numpy()))
        if not len(intersection):
            return 0
        intersection = np.array(list(intersection))
        allpairs = torch.vstack([
            allindices.t()[allindices[0] == index]
            for index in intersection
        ])
        # cannot index into a sparse tensor with pairs
        # so we have to find the indices of these pairs, and then
        # find the values corresponding to those.
        targets = torch.Tensor([
            self.co_occurrence[item[0]][item[1]]
            for item in allpairs
        ])
        weights_x = self._loss_weights(targets)
        i_indices = allpairs[:, 0]
        j_indices = allpairs[:, 1]
        current_weights = self._update(i_indices, j_indices)
        loss = weights_x * F.mse_loss(
            current_weights, torch.log(targets), reduction='none')
        # This is a feasible strategy for mapping indices -> pairs
        # Second strategy: Loop over all possible pairs
        # More computationally intensive
        # Allow this to be configurable?
        # Degrees of separation but only allow 1 or -1
        # We may want to save this loss as an attribute on the layer object
        # Does Lightning have a way of representing layer specific losses
        # Define an interface by which we can return loss objects
        # probably stick with self.losses = [loss]
        # - a list - because then we can do +=
        return torch.mean(loss).to(self.device)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # eventually, load this from a .pt file that contains all the
        # wi/wj/etc.
        raise NotImplementedError('Not yet implemented')


class GloveSimple(pl.LightningModule):

    def __init__(self, train_embeddings_file, batch_size, train_cooccurrence_file,
                 limit=None):
        # train_embeddings_file: the filename contaning the pre-trained weights
        # train_cooccurrence_file: the filename containing the co-occurrence statistics
        # that we want to match these embeddings to.
        super(GloveSimple, self).__init__()
        self.limit = limit
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
        self.glove_layer = GloveLayer(
        self.num_embeddings, self.embedding_dim,
        self.train_cooccurrence, device=self.device)

    def forward(self, indices):
        return self.glove_layer(indices)

    def training_step(self, batch, batch_idx):
        # input: indices of embedding
        # targets: target embeddings
        indices, targets = batch
        # Ideal outcome:
        # Pytorch object that is substitutable with
        # nn.Embedding
        # Subclass of nn.Embedding
        # but internally, we should give the option to use
        # a co-occurrence matrix
        out = self.glove_layer.weights[indices]
        loss = F.mse_loss(out, targets)
        loss += self.glove_layer.loss(indices)
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
        #opt = optim.Adagrad(self.parameters(), lr=0.05)
        opt = optim.Adam(self.parameters(), lr=0.005)
        return opt

    def train_dataloader(self):
        dataset = GloveEmbeddingsDataset(self.train_data, self.limit, self.device)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class GloveEmbeddingsDataset(Dataset):
    # Dataset for existing embedings

    def __init__(self, data, limit=None, device='cpu'):
        # train_data contains wi.weight / wj.weight / bi.weight / bj.weight
        # for stability, the target is the wi + wj
        self.weights = data['wi.weight'] + data['wj.weight']
        if limit:
            self.weights = self.weights[:limit]
        nemb, dim = self.weights.shape
        self.device = device
        self.x = torch.arange(nemb).to(self.device)
        self.y = self.weights.to(self.device)
        self.N = nemb

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        y = self.y[idx]
        return x, y


if __name__ == '__main__':

    model = GloveSimple('glove_audio.pt', batch_size=100,
                        train_cooccurrence_file='../audioset/co_occurrence_audio_all.pt')#, limit=1000)
    trainer = pl.Trainer(gpus=0, max_epochs=100, progress_bar_refresh_rate=20)
    trainer.fit(model)