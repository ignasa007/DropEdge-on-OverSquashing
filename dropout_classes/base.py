import torch.nn as nn


class BaseDropout(nn.Module):

    def __init__(self, dropout_prob=0.):

        super(BaseDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def apply_feature_mat(self, x, training=True):

        '''
        Dropout methods applied to the feature matrix, eg.
            1. Dropout

        Args:
            x (Tensor): feature matrix, eg. shape (|V|, D_{i})
        '''

        return x
    
    def apply_adj_mat(self, edge_index, training=True):

        '''
        Dropout methods applied to the adjacency matrix, eg.
            1. DropEdge
            2. DropNode

        Args:
            edge_index (Adj): adjacency matrix, eg. shape (2, |E|)
        '''

        return edge_index
    
    def apply_message_mat(self, messages, training=True):

        '''
        Dropout methods applied to the message matrix, eg.
            1. DropMessage

        Args:
            messages (Tensor): message matrix, eg. shape (|E|, D_{i+1})
        '''

        return messages