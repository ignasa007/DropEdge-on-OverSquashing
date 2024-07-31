import os
from glob import glob
import pickle
from tqdm import tqdm
import logging

import numpy as np
from torch_geometric.datasets import TUDataset

from utils.parse_logs import parse_configs
from utils.logger import get_time
from sensitivity.utils import *


try:

    model_dir = '..\\results\\model-store\\over-squashing'
    dir_names = {float(parse_configs(fn)['drop_p']): os.path.dirname(fn) for fn in glob(f'{model_dir}\\*\\logs')}
    ps = sorted(dir_names)
    assert ps[0] == 0.

    MOLECULE_SAMPLES = 100
    MODEL_SAMPLES = 10

    dataset = TUDataset(root='./data', name='Proteins', use_node_attr=True)
    num_nodes = np.array([molecule.num_nodes for molecule in dataset])

    logging.basicConfig(filename='tmp.log', level=logging.DEBUG)    
    results_dir = f'..\\results\\sensitivity\\{get_time()}'

    for _ in range(MOLECULE_SAMPLES):

        while True:
            i = np.random.choice(np.where(num_nodes <= 60)[0])
            molecule = dataset[i]
            try:
                commute_times = compute_commute_times(molecule.edge_index, P=0.).flatten()
            except AssertionError:
                continue
            shortest_distances = compute_shortest_distances(molecule.edge_index).flatten()
            break

        print(i, molecule.num_nodes)
        i_dir = os.path.join(results_dir, f'i={i}')
        os.makedirs(os.path.join(i_dir, 'jac-norms', 'untrained'), exist_ok=True)
        os.makedirs(os.path.join(i_dir, 'jac-norms', 'trained'), exist_ok=True)
        
        with open(os.path.join(i_dir, 'commute_times.pkl'), 'wb') as f:
            pickle.dump(commute_times, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(i_dir, 'shortest_distances.pkl'), 'wb') as f:
            pickle.dump(shortest_distances, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        for use_trained in (False, True):
            trained_sub_dir = 'trained' if use_trained else 'untrained'
            for p in tqdm(ps):
                jac_norms = get_jacobian_norms(molecule, dir_names[p], n_samples=MODEL_SAMPLES, use_trained=use_trained)
                with open(os.path.join(i_dir, 'jac-norms', trained_sub_dir, f'p={int(100*p)}.pkl'), 'wb') as f:
                    pickle.dump(jac_norms.flatten(), f, protocol=pickle.HIGHEST_PROTOCOL)

except Exception as e:
    
    logging.info(e)