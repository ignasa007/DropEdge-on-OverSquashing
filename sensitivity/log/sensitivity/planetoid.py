import os
from glob import glob
import pickle
from tqdm import tqdm

import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph

from utils.parse_logs import parse_configs
from sensitivity.utils import *
from sensitivity.utils.planetoid import *


DATASET = 'PubMed'
model_dir = f'./results/model-store/{DATASET}'
results_dir = f'./results/sensitivity/{DATASET}'

NODE_SAMPLES = 100
MODEL_SAMPLES = 10

dir_names = {float(parse_configs(fn)['drop_p']): os.path.dirname(fn) for fn in glob(f'{model_dir}/*/logs')}
Ps = sorted(dir_names)
assert Ps[0] == 0.

dataset = Planetoid(root='./data', name=DATASET)
num_nodes = dataset.x.size(0)
A = to_scipy_sparse_matrix(dataset.edge_index)
indices, = np.where(connected_components(A, return_labels=True)[1] == 0)
print(indices.shape)

for _ in range(NODE_SAMPLES):

    i = np.random.choice(indices)
    i_dir = os.path.join(results_dir, f'i={i}')
    os.makedirs(i_dir, exist_ok=True)
    print(i)

    shortest_distances = torch.from_numpy(shortest_path(A, method='D', indices=i))
    subset = torch.where(shortest_distances <= 4)[0]
    print(subset.size())
    edge_index, _ = subgraph(subset, dataset.edge_index, relabel_nodes=True, num_nodes=dataset.x.size(0))
    x = dataset.x[subset, :]
    new_i = torch.where(subset == i)[0].item()

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