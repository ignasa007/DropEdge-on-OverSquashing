import os
from glob import glob
import pickle
from tqdm import tqdm
import argparse

import numpy as np
from torch_geometric.datasets import TUDataset

from dataset.utils import normalize_features
from utils.parse_logs import parse_configs
from utils.format import format_dataset_name
from sensitivity.utils import *
from sensitivity.utils.tudataset import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--L', type=int)
args = parser.parse_args()

DATASET = args.dataset
L = args.L
model_dir = f'./results/model-store/{format_dataset_name[DATASET.lower()]}/L={L}'
results_dir = f'./results/sensitivity/{format_dataset_name[DATASET.lower()]}/L={L}'
os.makedirs(results_dir, exist_ok=True)

MOLECULE_SAMPLES = 100 - len(os.listdir(results_dir))
MODEL_SAMPLES = 10

dir_names = {float(parse_configs(fn)['drop_p']): os.path.dirname(fn) for fn in glob(f'{model_dir}/*/logs')}
Ps = sorted(dir_names)
assert Ps[0] == 0.

dataset = TUDataset(root='./data', name=DATASET, use_node_attr=True)
dataset, = normalize_features(dataset)
num_nodes = np.array([molecule.num_nodes for molecule in dataset])

for _ in range(MOLECULE_SAMPLES):

    while True:
        
        i = np.random.choice(np.where(num_nodes <= 60)[0])
        i_dir = os.path.join(results_dir, f'i={i}')
        if os.path.isdir(i_dir):
            continue
        
        molecule = dataset[i]
        try:
            commute_times = compute_commute_times(molecule.edge_index, P=0.).flatten()
        except AssertionError:
            continue
        shortest_distances = compute_shortest_distances(molecule.edge_index).flatten()
        break

    print(i, molecule.num_nodes)
    os.makedirs(os.path.join(i_dir, 'jac-norms', 'untrained'))
    os.makedirs(os.path.join(i_dir, 'jac-norms', 'trained'))
    
    with open(os.path.join(i_dir, 'commute_times.pkl'), 'wb') as f:
        pickle.dump(commute_times, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(i_dir, 'shortest_distances.pkl'), 'wb') as f:
        pickle.dump(shortest_distances, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    for use_trained in (False, True):
        trained_fn = 'trained' if use_trained else 'untrained'
        for P in tqdm(Ps):
            jac_norms = get_jacobian_norms(molecule, dir_names[P], n_samples=MODEL_SAMPLES, use_trained=use_trained)
            save_fn = os.path.join(i_dir, 'jac-norms', f'P={P}', f'{trained_fn}.pkl')
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
            with open(save_fn, 'wb') as f:
                pickle.dump(jac_norms.flatten(), f, protocol=pickle.HIGHEST_PROTOCOL)