'''
Poor naming of the file :(
    considers the sensitivity in the case the propagation matrix is D^{-1/2}AD^{-1/2}
'''


import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, add_self_loops, dropout_edge
import matplotlib.pyplot as plt

from sensitivity.utils import to_adj_mat, compute_shortest_distances, bin_jac_norms

L = 8
MOLECULE_SAMPLES = 100
DROPEDGE_SAMPLES = 20
dataset = TUDataset(root='./data', name='PROTEINS', use_node_attr=True)
indices = np.where(np.array([molecule.num_nodes for molecule in dataset]) <= 60)[0]
ps = np.arange(0, 1, 0.1)

sum_sensitivity = {p: torch.zeros(MOLECULE_SAMPLES, L+1) for p in ps}
count_sensitivity = {p: torch.zeros_like(sum_sensitivity[p]) for p in ps}

for m in range(MOLECULE_SAMPLES):

    while True:
        i = np.random.choice(indices)
        molecule = dataset[i]
        edge_index = molecule.edge_index
        try:
            shortest_distances = compute_shortest_distances(edge_index).flatten()
        except AssertionError:
            continue
        break

    # edge_index = torch.Tensor([[0, 1, 1, 1, 2, 2, 3, 3], [1, 0, 2, 3, 1, 3, 1, 2]]).type(torch.int64)
    x_sd = shortest_distances.unique().int()
    x_sd = x_sd[x_sd<=L]
    
    for p in ps:
            
        P_p = torch.zeros(molecule.num_nodes, molecule.num_nodes)
        for _ in range(DROPEDGE_SAMPLES):
            dropped_edge_index = add_self_loops(dropout_edge(edge_index, p, force_undirected=True)[0])[0]
            A = to_adj_mat(dropped_edge_index, num_nodes=molecule.num_nodes, assert_connected=False)
            degrees = degree(dropped_edge_index[0], num_nodes=molecule.num_nodes)
            deg_inv_sqrt = degrees.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            P_p += deg_inv_sqrt.unsqueeze(dim=0) * A * deg_inv_sqrt.unsqueeze(dim=1)
        P_p /= DROPEDGE_SAMPLES
        P_p_L = torch.matrix_power(P_p, L).flatten()
        
        y_sd = bin_jac_norms(P_p_L, shortest_distances, x_sd, agg='mean')
        sum_sensitivity[p][m, x_sd] += y_sd
        count_sensitivity[p][m, x_sd] += 1

for p in ps:

    # to avoid zero division error in case no graph hits shortest distance L
    dim_to_keep = (count_sensitivity[p]>0).any(dim=0)
    
    mean_sensitivity = sum_sensitivity[p][:, dim_to_keep].sum(dim=0) / count_sensitivity[p][:, dim_to_keep].sum(dim=0)
    plt.plot(mean_sensitivity, label=f'P = {p:.1f}')

plt.yscale('log')
plt.grid()
plt.legend()
plt.show()