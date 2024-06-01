from argparse import Namespace
from typing import Union

from torch import Tensor, BoolTensor
from torch.nn import Module, Sequential
from torch_geometric.typing import Adj

from model.gnn import get_activation, get_layer
from model.ffn import get_head
from dropout import get_dropout


class Model(Module):

    def __init__(self, input_dim: int, num_classes: int, args: Namespace):
        
        super(Model, self).__init__()
        
        drop_strategy = get_dropout(args.dropout)(args.dropout_prob)
        gnn_layer = get_layer(gnn_layer)
        gnn_layer_sizes = [input_dim] + args.gnn_layer_sizes
        module_list = []
        for in_channels, out_channels in zip(gnn_layer_sizes[:-1], gnn_layer_sizes[1:]):
            module_list.append(gnn_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                drop_strategy=drop_strategy,
                activation=get_activation(args.gnn_activation)(),
                add_self_loops=args.add_self_loops,
                normalize=args.normalize,
            ))
        self.message_passing = Sequential(*module_list)

        ffn_head = get_head(args.task)
        ffn_layer_sizes = args.gnn_layer_sizes[-1:] + args.ffn_layer_sizes
        self.readout = ffn_head(
            layer_sizes=ffn_layer_sizes,
            num_classes=num_classes,
            activation=get_activation(args.ffn_activation)(),
            task=args.task,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        target: Tensor,
        mask: Union[Tensor, BoolTensor, None] = None,
    ):

        node_embeddings = self.message_passing(x, edge_index)
        loss = self.readout(node_embeddings, target, mask)

        return loss
    
    def compute_metrics(self):

        return self.readout.metrics.compute()