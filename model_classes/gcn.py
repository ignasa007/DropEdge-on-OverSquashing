from torch_geometric.nn.conv import GCNConv

'''
Graph Drop Connect needs an implementation of the aggregation function (after the message step)
'''

class GCNLayer(GCNConv):

    def __init__(self):

        super(GCNLayer, self).__init__()

    def __call__(self, x, edge_index):

        '''
        Drop whatever you want from the input:
            1. Dropout
                https://github.com/zjunet/DropMessage/blob/1b52da82daf52a426fb7364fea60eb90b38d0b8b/src/layer.py#L31
            2. DropNode
                https://github.com/zjunet/DropMessage/blob/1b52da82daf52a426fb7364fea60eb90b38d0b8b/src/layer.py#L21
            3. DropEdge -- perturb the adjacency matrix once in each forward pass?
                https://github.com/zjunet/DropMessage/blob/1b52da82daf52a426fb7364fea60eb90b38d0b8b/src/layer.py#L24
        '''
        
        pass

    def message(self):

        '''
        Drop whatever you want from the message matrix:
            1. DropMessage
        '''

        pass