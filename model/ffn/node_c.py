from typing import Union
from torch import Tensor, BoolTensor
from model.ffn.base import BaseHead


class NodeClassification(BaseHead):

    def __init__(self, layer_sizes, activation):

        super(NodeClassification, self).__init__(
            task='classification',
            layer_sizes=layer_sizes,
            activation=activation
        )

    def forward(self, node_repr: Tensor, target: Tensor, mask: Union[BoolTensor, None] = None):

        if mask is not None:
            node_repr = node_repr[mask, :]
            target = target[mask, :]

        logits = self.ffn(node_repr)
        loss = self.metrics.update(logits, target)

        return loss