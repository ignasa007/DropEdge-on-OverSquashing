from torch_geometric.utils import dropout_edge
from .base import BaseDropout


class DropEdge(BaseDropout):

    def __init__(self, dropout_prob=0.):

        super(DropEdge, self).__init__(dropout_prob)

    def apply_adj_mat(self, edge_index, training=True):

        '''
        when we have edge attributes, will need
            https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_adj
        ''' 
        
        return dropout_edge(edge_index, p=self.dropout_prob, training=training)[0]