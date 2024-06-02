from typing import Union
from argparse import Namespace

from torch import Tensor
from torch.nn import Module
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import GCNConv

from dropout.base import BaseDropout
from model.message_passing.pretreatment import ModelPretreatment


# TODO: Graph Drop Connect needs an implementation of the aggregation function (after the message step)

class GCNLayer(GCNConv):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        activation: Module,
        args: Namespace,
    ):

        super(GCNLayer, self).__init__(
            in_channels,
            out_channels,
            add_self_loops=args.add_self_loops,
            normalize=args.normalize
        )
        
        self.pt = ModelPretreatment(args.add_self_loops, args.normalize)
        self.activation = activation
        self.drop_strategy = drop_strategy
        
    def forward(self, x: Tensor, edge_index: Adj):

        # FEATURE TRANSFORMATION
        # drop from feature matrix -- dropout and drop node
        x = self.drop_strategy.apply_feature_mat(x, self.training)
        x = self.lin(x)
        x = self.activation(x)

        # MESSAGE PASSING
        # drop from adj matrix -- drop edge, drop gnn and drop agg -- and normalize it
        edge_index, _ = self.drop_strategy.apply_adj_mat(edge_index, None, self.training)
        edge_index, edge_weight = self.pt.pretreatment(x.size(0), edge_index, x.dtype)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor):

        if edge_weight is not None:
            x_j = x_j * edge_weight.view(-1, 1)

        # drop from message matrix -- drop message
        x_j = self.drop_strategy.apply_message_mat(x_j, self.training)

        return x_j