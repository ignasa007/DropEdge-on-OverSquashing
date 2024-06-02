from typing import Union
from torch import Tensor, BoolTensor
from model.readout.base import BaseHead


class NodeRegression(BaseHead):

    def preprocess(self, node_repr: Tensor, mask: Union[BoolTensor, None] = None):

        '''
        Preprocess the input -- filter out the masked nodes' embeddings.

        Args:
            node_repr: tensor of shape (N, H), where $N is the number of nodes in the graph, and
                $H is the dimension of messages.
            mask: boolean tensor of shape (N,) indicating which nodes to compute metrics over.
        '''

        if mask is not None:
            node_repr = node_repr[mask, ...]

        return node_repr