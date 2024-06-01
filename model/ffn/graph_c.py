from typing import Union
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from model.ffn.base import BaseHead


class GraphClassification(BaseHead):

    def __init__(self, layer_sizes, activation):

        super(GraphClassification, self).__init__(
            task_name='classification',
            layer_sizes=layer_sizes,
            activation=activation
        )

    def preprocess(self, node_repr: Tensor, target: Tensor, mask: Union[Tensor, None] = None):

        '''
        Preprocess the input -- compute the mean of the node embeddings from each graph.

        Args:
            node_repr: tensor of shape (N_1+...+N_B, H), where $N_i is the number of nodes in graph $i,
                $B is the batch size, and $H is the dimension of messages.
            target: true labels.
            mask: tensor (N_1, N_2, ..., N_B) of shape (B,)
        '''

        return global_mean_pool(x=node_repr, batch=mask), target