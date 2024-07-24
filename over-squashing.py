import os
from glob import glob
from collections import defaultdict
import pickle
from tqdm import tqdm

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
import torch
from torch.func import jacrev
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, to_undirected
import matplotlib.pyplot as plt

from model import Model as Base
from utils.parse_logs import parse_configs


def compute_commute_times(adj, P=0.):

    adj = to_undirected(adj.type(torch.int64))
    degrees = degree(adj[0], num_nodes=adj.max()+1)
    A = torch.zeros((degrees.size(0), degrees.size(0)))
    A[adj[0], adj[1]] = 1.
    assert connected_components(A, directed=False, return_labels=False) == 1

    L = torch.diag(degrees) - A
    L_pinv = torch.linalg.pinv(L)
    L_pinv_diag = torch.diag(L_pinv)
    
    beta = torch.sum(degrees / (1 - P**degrees))
    C = beta * (L_pinv_diag.unsqueeze(0) + L_pinv_diag.unsqueeze(1) - 2*L_pinv)

    return C


def compute_shortest_distances(adj):

    adj = to_undirected(adj.type(torch.int64))
    degrees = degree(adj[0], num_nodes=adj.max()+1)
    A = torch.zeros((degrees.size(0), degrees.size(0)))
    A[adj[0], adj[1]] = 1.
    assert connected_components(A, directed=False, return_labels=False) == 1

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
        mean = torch.mean(jac_norms[torch.where(bin_assignments == bin)])
        means.append(mean)

    return torch.Tensor(means)


def plot(x, y1, y2):

    plt.plot(x, y1, label='DropEdge')
    plt.plot(x, y2, label='NoDrop')
    plt.xlim(0, x[max(torch.where(y1!=0.)[0][-1], torch.where(y2!=0.)[0][-1])])
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()


def crossover_point(x, y1, y2):

    '''expects y1 to start above y2 but y1 to end below y2'''

    indices, = torch.where(y1 > y2)
    idx = indices[-1] if indices.numel() != 0 else 0

    if not (indices.numel() == 0 or torch.all(y1[:idx+1] >= y2[:idx+1])) or not torch.all(y1[idx+1:] <= y2[idx+1:]):
        plot(x, y1, y2)

    return x[idx]


try:

    ### GLOBAL CONSTANTS ###

    results_dir = '.\\results\\model-store\\over-squashing'
    dir_names = {float(parse_configs(fn)['drop_p']): os.path.dirname(fn) for fn in glob(f'{results_dir}\\*\\logs')}
    ps = sorted(dir_names)
    assert ps[0] == 0.

    MOLECULE_SAMPLES = 10
    MODEL_SAMPLES = 10
    BIN_SIZE = 40.

    dataset = TUDataset(root='./data', name='Proteins', use_node_attr=True)
    num_nodes = np.array([molecule.num_nodes for molecule in dataset])

    ### START EXPERIMENT ###

    import logging
    logging.basicConfig(filename='tmp.log', level=logging.DEBUG)
    
    results = {distance_metric: defaultdict(list) for distance_metric in ('commute-times', 'shortest-distances')}

    for _ in range(MOLECULE_SAMPLES):

        ### SAMPLE MOLECULE ###

        while True:
            i = np.random.choice(np.where((30 <= num_nodes) & (num_nodes <= 70))[0])
            molecule = dataset[i]
            try:
                commute_times = compute_commute_times(molecule.edge_index, P=0.)
            except AssertionError:
                continue
            shortest_distances = compute_shortest_distances(molecule.edge_index)
            break

        print(i, molecule.num_nodes)

        ### COMMUTE TIMES ###

        binned_commute_times = torch.round(commute_times/BIN_SIZE) * BIN_SIZE
        binned_commute_times = binned_commute_times.flatten()
        x_ct = binned_commute_times.unique()

        ### SHORTEST DISTANCE ###

        shortest_distances = shortest_distances.flatten()
        x_sd = shortest_distances.unique()

        for use_trained in (False, True):

            ### JAC-NORMS FOR NO-DROP MODEL ###

            no_drop_jac_norms = get_jacobian_norms(molecule, dir_names[0.], n_samples=MODEL_SAMPLES, use_trained=use_trained)
            y_ct_no_drop = bin_jac_norms(no_drop_jac_norms, binned_commute_times, x_ct)
            y_sd_no_drop = bin_jac_norms(no_drop_jac_norms, shortest_distances, x_sd)
            
            for p in tqdm(ps[1:]):

                ### JAC-NORMS FOR DROP MODEL ###

                drop_jac_norms = get_jacobian_norms(molecule, dir_names[p], n_samples=MODEL_SAMPLES, use_trained=use_trained)
                y_ct_drop = bin_jac_norms(drop_jac_norms, binned_commute_times, x_ct)
                y_sd_drop = bin_jac_norms(drop_jac_norms, shortest_distances, x_sd)
                
                ### FIND CROSSOVER POINTS ###

                cp_ct = crossover_point(x_ct, y_ct_drop, y_ct_no_drop)
                cp_sd = crossover_point(x_sd, y_sd_drop, y_sd_no_drop)

                results['commute-times'][(p, use_trained)].append(cp_ct)
                results['shortest-distances'][(p, use_trained)].append(cp_sd)


    ### COMPUTE STATS OVER GRAPHS ###

    for distance_metric in ('commute-times', 'shortest-distances'):
        for k, v in results[distance_metric].items():
            results[distance_metric][k] = (np.mean(v), np.std(v))
        results[distance_metric] = dict(results[distance_metric])

    with open('results/crossover-points.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

except Exception as e:
    
    logging.info(e)