from typing import Union
from torch import Tensor, BoolTensor
from model.ffn.base import BaseHead


class NodeRegression(BaseHead):

    def __init__(self, layer_sizes, activation):

        super(NodeRegression, self).__init__(
            task_name='regression',
            layer_sizes=layer_sizes,
            activation=activation
        )

    def preprocess(self, node_repr: Tensor, target: Tensor, mask: Union[BoolTensor, None] = None):

        '''
        Preprocess the input -- compute the mean of the node embeddings from each graph.

        Args:
            node_repr: tensor of shape (N, H), where $N is the number of nodes in the graph, and
                $H is the dimension of messages.
            target: true labels.
            mask: boolean tensor of shape (N,) indicating which nodes to compute metrics over.
        '''

        if mask is not None:
            node_repr = node_repr[mask, ...]
            target = target[mask, ...]

        return node_repr, target