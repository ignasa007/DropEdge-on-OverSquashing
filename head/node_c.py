from typing import Union
from torch import Tensor, BoolTensor
from .base import BaseHead


class NodeClassification(BaseHead):

    def __init__(self):

        super(NodeClassification, self).__init__(task='classification')

    def forward(self, node_repr: Tensor, labels: Tensor, mask: Union[BoolTensor, None] = None):

        if mask is not None:
            node_repr = node_repr[mask]
            labels = labels[mask]

        result = self.metrics(node_repr, labels)

        return result