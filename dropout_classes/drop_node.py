from torch_geometric.utils import dropout_node
from .base import BaseDropout


class DropNode(BaseDropout):

    def __init__(self, dropout_prob=0.):

        super(DropNode, self).__init__(dropout_prob)
    
    def apply_feature_mat(self, x, training=True):

        return super(DropNode, self).apply_feature_mat(x, training)
    
    def apply_adj_mat(self, edge_index, training=True):
        
        return dropout_node(edge_index, p=self.dropout_prob, training=training)[0]
    
    def apply_message_mat(self, messages, training=True):

        return super(DropNode, self).apply_message_mat(messages, training)