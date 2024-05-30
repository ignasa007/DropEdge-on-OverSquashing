from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_sparse import SparseTensor, set_diag


class ModelPretreatment:
    
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def pretreatment(self, num_nodes: int, edge_index: Adj, dtype):

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                row, col = edge_index
            elif isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
            deg = degree(col, num_nodes, dtype=dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, edge_weight