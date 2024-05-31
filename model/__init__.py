from argparse import Namespace
from typing import Union

from torch import Tensor, BoolTensor
from torch.nn import Module, Sequential

from model.gnn import get_activation, get_layer
from model.ffn import BaseHead, get_head
from dropout import get_dropout


class Model(Module):

    def __init__(self, input_dim: int, output_dim: int, args: Namespace):

        # TODO: resolve the argument output_dim -- how to receive for regression tasks?
        # REMINDER: num_classes = 2 => model head has a single output; loss is binary cross-entropy
        #           num_classes > 2 => model has $num_classes outputs; loss is cross-entropy
        
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
        ffn_layer_sizes = args.gnn_layer_sizes[-1:] + args.ffn_layer_sizes + [output_dim]
        self.readout = ffn_head(
            layer_sizes=ffn_layer_sizes,
            activation=get_activation(args.ffn_activation)(),
        )

    def forward(
        self,
        *inputs,
        target: Tensor,
        mask: Union[BoolTensor, None] = None
    ):

        # TODO: check what the input format should be graph level tasks
        node_embeddings = self.message_passing(*inputs)
        loss = self.readout(node_embeddings, target, mask)

        return loss
    
    def compute_metrics(self):

        return self.readout.metrics.compute()