from torch_geometric.utils import drop_edge
from .base import BaseDropout


class DropNode(BaseDropout):

    def __init__(self, dropout_prob=0.):

        super(DropNode, self).__init__(dropout_prob)

    def apply_adj_mat(self, edge_index, training=True):
        
        return drop_edge(edge_index, p=self.dropout_prob, training=training)[0]