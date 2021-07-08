# Given a co-occurrence matrix, and a file of labels to names,
# and a file of the universe of labels to names,
# test that the correct indexes are included in the output
import sys

from spond.experimental.glove.aligned_glove import DataDictionary

sys.path.append("/opt/github.com/spond")

from io import StringIO
from unittest import TestCase
import torch

# all labels: These will be numbered 0 to 7

ALL_LABELS = f"""
#LabelName,DisplayName
/m/09x0r,"Speech"
/m/05zppz,"Male person"
/m/02zsn,"Female person"
/m/0ytgt,"Child"
/m/01h8n0,"Conversation"
/m/02qldy,"Narration, monologue"
/m/0261r1,"Babbling"
/m/0brhx,"Speech synthesizer"
"""

ALL_LABELS_INDEXES = list(range(8))

X_LABELS = f"""
mid,display_name
/m/09x0r,"Speech"
/m/05zppz,"Male person"
/m/02zsn,"Female person"
/m/0ytgt,"Child"
/m/01h8n0,"Conversation"
"""

X_LABELS_INDEXES = [0, 1, 2, 3, 4]

X_COOC = torch.tensor([
    [0.0, 2, 3, 3, 1],
    [2,   0, 4, 0, 3],
    [3,   4, 0, 2, 2],
    [3,   0, 2, 0, 1],
    [1,   3, 2, 1, 0],
])

Y_LABELS = f"""
mid,display_name
/m/05zppz,"Male person"
/m/02zsn,"Female person"
/m/0ytgt,"Child"
/m/02qldy,"Narration, monologue"
/m/0brhx,"Speech synthesizer"
"""

Y_LABELS_INDEXES = [1, 2, 3, 5, 7]

Y_COOC = torch.tensor([
    [0.0, 1, 5, 3, 1],
    [1,   0, 2, 0, 3],
    [5,   2, 0, 2, 1],
    [3,   0, 2, 0, 1],
    [1,   3, 1, 1, 0],
])

# first item: index in all_labels
# second item: index in x
# second item: index in y
UNION_INDEXES = [
    [1, 1, 0],
    [2, 2, 1],
    [3, 3, 2]
]


class TestCoocIndexParsing(TestCase):

    def setUp(self):
        self.all_labels = StringIO(ALL_LABELS.strip())
        self.all_labels_indexes = torch.tensor(ALL_LABELS_INDEXES)
        self.x_labels = StringIO(X_LABELS.strip())
        self.x_labels_indexes = torch.tensor(X_LABELS_INDEXES)
        self.y_labels = StringIO(Y_LABELS.strip())
        self.y_labels_indexes = torch.tensor(Y_LABELS_INDEXES)
        self.union_indexes = torch.tensor(UNION_INDEXES)

    def test(self):
        # all_labels, all_names = readlabels(self.all_labels, rootdir=None)
        # name_to_label = {v: k for k, v in all_names.items()}
        # index_to_label = {v: k for k, v in all_labels.items()}
        # index_to_name = {v: all_names[k] for k, v in all_labels.items()}
        # name_to_index = {v: k for k, v in index_to_name.items()}
        #
        # # x_labels is dictionary of label to index
        # # x_names is dictionary of label to name
        # x_labels, x_names = readlabels(self.x_labels, rootdir=None)
        # # same applies to y_labels and y_names
        # y_labels, y_names = readlabels(self.y_labels, rootdir=None)
        # y_name_to_label = {v: k for k, v in y_names.items()}
        # union = {}
        # for x_label, x_name in x_names.items():
        #     if x_label in y_names:
        #         union[x_name] = (x_labels[x_label], y_labels[y_name_to_label[x_name]])
        # self.union_names = union
        # # Tuple of (index in all, index in x, index in y)
        # # TODO: ugh, too many levels of indirection, clean up later.
        # self.union_indexes = torch.tensor([
        #     (
        #         name_to_index[name],
        #         x_labels[name_to_label[name]],
        #         y_labels[name_to_label[name]],
        #     )
        #     for name in union
        # ])

        dd = DataDictionary(
            torch.tensor(X_COOC),      # co-occurrence matrix for x
            self.x_labels,    # full filepath or handle containing all x labels to names
            torch.tensor(Y_COOC),      # co-occurrence matrix for y
            self.y_labels,    # full filepath or handle containing all y labels to names
            self.all_labels,
        )

        self.assertTrue((dd.x_indexes == self.x_labels_indexes).all().item())
        self.assertTrue((dd.y_indexes == self.y_labels_indexes).all().item())
        self.assertTrue((dd.union_indexes == self.union_indexes).all().item())
