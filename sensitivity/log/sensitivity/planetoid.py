import os
from glob import glob
import pickle
from tqdm import tqdm
import argparse

import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph

from utils.parse_logs import parse_configs
from utils.format import format_dataset_name
from sensitivity.utils import *
from sensitivity.utils.planetoid import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--L', type=int)
args = parser.parse_args()

DATASET = args.dataset
L = args.L
model_dir = f'./results/model-store/{format_dataset_name[DATASET.lower()]}/L={L}'
results_dir = f'./results/sensitivity/{format_dataset_name[DATASET.lower()]}/L={L}'
os.makedirs(results_dir, exist_ok=True)

NODE_SAMPLES = 100 - len(os.listdir(results_dir))
MODEL_SAMPLES = 10

dir_names = {float(parse_configs(fn)['drop_p']): os.path.dirname(fn) for fn in glob(f'{model_dir}/*/logs')}
Ps = sorted(dir_names)
assert Ps[0] == 0.

dataset = Planetoid(root='./data', name=DATASET, split='full')
num_nodes = dataset.x.size(0)
A = to_scipy_sparse_matrix(dataset.edge_index)

# sample nodes from the largest component
assignments = connected_components(A, return_labels=True)[1]
cc_labels, sizes = np.unique(assignments, return_counts=True)
indices = np.where(assignments == cc_labels[np.argmax(sizes)])[0]

for _ in range(NODE_SAMPLES):

    while True:
        i = np.random.choice(indices)
        i_dir = os.path.join(results_dir, f'i={i}')
        if not os.path.isdir(i_dir):
            break
    os.makedirs(i_dir)

    shortest_distances = torch.from_numpy(shortest_path(A, method='D', indices=i))
    subset = torch.where(shortest_distances <= L)[0]
    edge_index, _ = subgraph(subset, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.x.size(0))
    # checked implementation and relabelling is such that subset[i] is relabelled as i
    x = dataset.x[subset, :]
    new_i = torch.where(subset == i)[0].item()

    print(i, subset.size(0))

    with open(os.path.join(i_dir, 'shortest_distances.pkl'), 'wb') as f:
        pickle.dump(shortest_distances[subset], f, protocol=pickle.HIGHEST_PROTOCOL)
    
    for use_trained in (False, True):
        trained_fn = 'trained' if use_trained else 'untrained'
        for P in tqdm(Ps):
            jac_norms = get_jacobian_norms(x, edge_index, new_i, dir_names[P], MODEL_SAMPLES, use_trained)
            save_fn = os.path.join(i_dir, 'jac-norms', f'P={P}', f'{trained_fn}.pkl')
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
            with open(save_fn, 'wb') as f:
                pickle.dump(jac_norms.flatten(), f, protocol=pickle.HIGHEST_PROTOCOL)