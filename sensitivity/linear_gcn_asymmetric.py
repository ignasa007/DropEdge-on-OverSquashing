'''
Poor naming of the file :(
    considers the sensitivity in the case the propagation matrix is D^{-1}A
'''


import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import matplotlib.pyplot as plt

from sensitivity.utils import to_adj_mat, compute_shortest_distances, bin_jac_norms

L = 8
MOLECULE_SAMPLES = 100
dataset = TUDataset(root='./data', name='PROTEINS', use_node_attr=True)
indices = np.where(np.array([molecule.num_nodes for molecule in dataset]) <= 60)[0]
ps = np.arange(0, 1, 0.1)

sum_sensitivity = {p: torch.zeros(MOLECULE_SAMPLES, L+1) for p in ps}
count_sensitivity = {p: torch.zeros_like(sum_sensitivity[p]) for p in ps}

for m in range(MOLECULE_SAMPLES):

    while True:
        i = np.random.choice(indices)
        edge_index = dataset[i].edge_index
        try:
            shortest_distances = compute_shortest_distances(edge_index).flatten()
        except AssertionError:
            continue
        break

    # edge_index = torch.Tensor([[0, 1, 1, 1, 2, 2, 3, 3], [1, 0, 2, 3, 1, 3, 1, 2]]).type(torch.int64)
    degrees = degree(edge_index[0])
    A = to_adj_mat(edge_index, assert_connected=False)
    x_sd = shortest_distances.unique().int()
    x_sd = x_sd[x_sd<=L]
    
    for p in ps:
            
        c_p = (1-p**(degrees+1)) / (1-p)
        if p == 0:
            non_diag = 1 / (degrees+1)
            diag = 1 / (degrees+1)
        else:
            non_diag = (1/degrees - c_p/(degrees*(degrees+1)))
            diag = (1-p**(degrees+1)) / ((1-p)*(degrees+1))
        
        non_diag = non_diag.unsqueeze(dim=1).repeat(1, degrees.size(0)) * A
        diag = torch.diag(diag)
        P_p = torch.where(diag>0., diag, non_diag)
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