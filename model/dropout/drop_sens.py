import torch
from torch_geometric.utils import degree, remove_self_loops
from model.dropout.base import BaseDropout


class DropSens(BaseDropout):

    def __init__(self, dropout_prob=0.5):

        super(DropSens, self).__init__(dropout_prob)
    
    def apply_feature_mat(self, x, training=True):

        return super(DropSens, self).apply_feature_mat(x, training)
    
    def apply_adj_mat(self, edge_index, edge_attr=None, training=True):

        if not training or self.dropout_prob == 0.0:
            return edge_index, edge_attr
        
        # removing self-loops here because relevant to degree computation
        # didn't remove in DropEdge because self-loops are added back in pretreatment
        edge_index, _ = remove_self_loops(edge_index)

        if not hasattr(self, 'q_d'):
            degrees = degree(edge_index[1])
            self.q_d = self.dropout_prob ** (1/degrees[edge_index[1]])
        # print(edge_index[1][:10], degrees[:10], degrees[edge_index[1]][:10], self.q_d[:10], sep='\n'); import sys; sys.exit()
        edge_mask = torch.bernoulli(self.q_d).bool()
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        return edge_index, edge_attr
    
    def apply_message_mat(self, messages, training=True):

        return super(DropSens, self).apply_message_mat(messages, training)