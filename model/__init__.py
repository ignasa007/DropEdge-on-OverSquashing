from torch import Tensor
from torch_geometric.typing import Adj
from torch.nn import Module, ModuleList
from .maps import get_activation, get_layer
from ..dropout import get_dropout


class Model(Module):

    def __init__(
        self,
        input_dim: int,
        h_layer_sizes: list,
        output_dim: int,
        gnn_layer: str = 'gcn',
        add_self_loops: bool = True,
        normalize: bool = True,
        drop_strategy: str = 'dropout',
        dropout_prob: float = 0.5,
        activation: str = 'relu',
    ):

        super(Model, self).__init__()

        drop_strategy = get_dropout(drop_strategy)(dropout_prob)
        gnn_layer = get_layer(gnn_layer)

        h_layer_sizes = [input_dim] + h_layer_sizes
        self.module_list = ModuleList([
            gnn_layer(drop_strategy, in_channels, out_channels, add_self_loops, normalize)
            for in_channels, out_channels in zip(h_layer_sizes[:-1], h_layer_sizes[1:])
        ])
        self.n_layers = len(h_layer_sizes) - 1
        
        self.activation = get_activation(activation)()

        # TODO: implement a classifier head (will use h_layer_sizes[-1] and output_dim)

    def forward(self, x: Tensor, edge_index: Adj):

        # TODO: check if setting training=True/False of this class sets training of the layers

        for i, gnn_layer in enumerate(self.module_list, 1):
            x = gnn_layer(x, edge_index)
            if i != self.n_layers:
                x = self.activation(x)

        # TODO: after message passing, need a classifier head to get the final output