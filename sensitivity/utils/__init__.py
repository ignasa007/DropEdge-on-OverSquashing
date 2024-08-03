from scipy.sparse.csgraph import connected_components, shortest_path

import torch
from torch_geometric.utils import degree, to_undirected


def to_adj_mat(edge_index, assert_connected=True):

    edge_index = to_undirected(edge_index)
    num_nodes = (edge_index.max()+1).item()
    A = torch.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1.

    if assert_connected:
        assert connected_components(A, directed=False, return_labels=False) == 1

    return A


def compute_commute_times(edge_index, P=0.):

    edge_index = edge_index.type(torch.int64)
    A = to_adj_mat(edge_index, assert_connected=True)
    degrees = degree(edge_index[0], num_nodes=edge_index.max()+1)

    L = torch.diag(degrees) - A
    L_pinv = torch.linalg.pinv(L)
    L_pinv_diag = torch.diag(L_pinv)
    
    beta = torch.sum(degrees / (1 - P**degrees))
    C = beta * (L_pinv_diag.unsqueeze(0) + L_pinv_diag.unsqueeze(1) - 2*L_pinv)

    return C


def compute_shortest_distances(edge_index, assert_connected=True):

    A = to_adj_mat(edge_index, assert_connected=assert_connected)
    shortest_distances = torch.from_numpy(shortest_path(A.numpy(), directed=False))
    
    return shortest_distances


def bin_jac_norms(jac_norms, bin_assignments, bins):

    if jac_norms.ndim > 1:
        jac_norms = jac_norms.flatten()
    
    assert jac_norms.size() == bin_assignments.size()

    means = list()
    for bin in bins:
        bin_members = jac_norms[torch.where(bin_assignments == bin)]
        means.append(torch.mean(bin_members))

    return torch.Tensor(means)