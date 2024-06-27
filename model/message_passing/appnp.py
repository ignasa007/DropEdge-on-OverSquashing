from argparse import Namespace

from torch.nn import Module

from dropout.base import BaseDropout
from model.message_passing.gcn import GCNLayer


class APPNPLayer(GCNLayer):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_strategy: BaseDropout,
        activation: Module,
        config: Namespace,
    ):

        super(APPNPLayer, self).__init__(
            in_channels,
            out_channels,
            drop_strategy,
            activation,
            config
        )

        self.power_iter = config.power_iter
        self.teleport_p = config.teleport_p

    def message_passing(self, edge_index, x, edge_weight):

        h = x
        for _ in range(self.power_iter):
            x = (1-self.teleport_p) * self.propagate(edge_index, x=x, edge_weight=edge_weight) \
                +  self.teleport_p  * h
            
        return x