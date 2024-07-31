import pickle
from scipy.sparse.csgraph import connected_components, shortest_path

import torch
from torch.func import jacrev
from torch_geometric.utils import degree, to_undirected

from model import Model as Base


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


def compute_shortest_distances(edge_index):

    A = to_adj_mat(edge_index, assert_connected=True)
    shortest_distances = torch.from_numpy(shortest_path(A.numpy(), directed=False))
    
    return shortest_distances


class Model(Base):
    
    def forward(self, edge_index, mask, x):
    
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)
    
        return x
    

def get_jacobian_norms(molecule, dir_name, n_samples, use_trained):

    with open(f'{dir_name}\\config.pkl', 'rb') as f:
        config = pickle.load(f)

    model = Model(config)
    if use_trained:
        state_dict = torch.load(f'{dir_name}\\ckpt-400.pt')
        model.load_state_dict(state_dict)
    model.train()

    jacobians = torch.zeros((molecule.num_nodes, config.gnn_layer_sizes[-1], molecule.num_nodes, config.input_dim))
    n_samples = n_samples if config.drop_p > 0. else 1
    for _ in range(n_samples):
        jacobians += jacrev(model, argnums=2)(molecule.edge_index, None, molecule.x)
    jacobians /= n_samples
    jacobian_norms = jacobians.transpose(1, 2).flatten(start_dim=2).norm(dim=2, p=1)

    return jacobian_norms


def bin_jac_norms(jac_norms, bin_assignments, bins):

    if jac_norms.ndim > 1:
        jac_norms = jac_norms.flatten()
    
    assert jac_norms.size() == bin_assignments.size()

    means = list()
    for bin in bins:
        bin_members = jac_norms[torch.where(bin_assignments == bin)]
        means.append(torch.mean(bin_members))

    return torch.Tensor(means)