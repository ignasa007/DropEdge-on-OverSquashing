'''
Poor naming of the file :(
    considers the sensitivity in the case the propagation matrix is D^{-1/2}AD^{-1/2}
'''

import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, add_self_loops, dropout_edge
import matplotlib.pyplot as plt

from sensitivity.utils import to_adj_mat, compute_shortest_distances, bin_jac_norms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Proteins', 'MUTAG'])
args = parser.parse_args()

L = 12
MOLECULE_SAMPLES = 100
DROPEDGE_SAMPLES = 10
dataset = TUDataset(root='./data/TUDataset', name=args.dataset.upper(), use_node_attr=True)
indices = np.where(np.array([molecule.num_nodes for molecule in dataset]) <= 60)[0]
ps = np.arange(0, 1, 0.1)[::2]

sum_sensitivity = {p: torch.zeros(MOLECULE_SAMPLES, L+1) for p in ps}
count_sensitivity = {p: torch.zeros_like(sum_sensitivity[p]) for p in ps}
fig, ax = plt.subplots(1, 1)

for m in tqdm(range(MOLECULE_SAMPLES)):

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
        
            dropped_edge_index = add_self_loops(dropout_edge(edge_index, p, force_undirected=False)[0])[0]
            A = to_adj_mat(dropped_edge_index, num_nodes=molecule.num_nodes, assert_connected=False)
        
            in_deg_inv = degree(dropped_edge_index[1], num_nodes=molecule.num_nodes).pow(-1)
            in_deg_inv[in_deg_inv == float('inf')] = 0
        
            P_p += torch.diag(in_deg_inv) @ A
        
        P_p /= DROPEDGE_SAMPLES
        P_p_L = torch.matrix_power(P_p, L).flatten()
        
        y_sd = bin_jac_norms(P_p_L, shortest_distances, x_sd, agg='mean')
        sum_sensitivity[p][m, x_sd] += y_sd
        count_sensitivity[p][m, x_sd] += 1

for p in ps:

    # to avoid zero division error in case no graph hits shortest distance L
    dim_to_keep = (count_sensitivity[p]>0).any(dim=0)
    x = torch.where(dim_to_keep)[0]
    y = sum_sensitivity[p][:, dim_to_keep].sum(dim=0) / count_sensitivity[p][:, dim_to_keep].sum(dim=0)
    ax.plot(x, y, label=f'P = {p:.1f}')

ax.set_yscale('log')
ax.set_xlabel('Shortest Distance', fontsize=14)
ax.set_ylabel(rf'$\left(\mathbb{{E}}\left[\hat{{A}}^{{asym}}\right]^{{{L}}}\right)_{{ij}}$', fontsize=14)
ax.set_title(args.dataset, fontsize=14)
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor = (0, -0.07, 1, 1))
fig.tight_layout()
fn = f'./assets/linear-gcn/asymmetric_emp/{args.dataset}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')