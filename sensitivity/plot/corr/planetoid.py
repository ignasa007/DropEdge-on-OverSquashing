import pickle
import argparse

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from sensitivity.utils import bin_jac_norms
from utils.format import format_dataset_name


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer'])
parser.add_argument('--L', type=int)
parser.add_argument('--agg', type=str, default='mean', choices=['mean', 'sum'])
args = parser.parse_args()

DATASET = args.dataset
L = args.L
agg = args.agg
RESULTS_DIR = f'./results/sensitivity/{format_dataset_name[DATASET.lower()]}/L={L}'

with open(f'{RESULTS_DIR}/indices.pkl', 'rb') as f:
    indices: list = pickle.load(f)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f'{DATASET}, L={L}')
metric_names = ('train_ce', 'train_mae', 'eval_ce', 'eval_mae')

for P in np.round(np.arange(0.0, 1.0, 0.2), decimals=1):

    # 2nd dim = L+1 because we trained a L-layer GNN, so it can reach nodes at distances from 0 and L 
    binned_jac_norms = torch.full((len(indices), L+1), torch.nan)
    for i, idx in enumerate(indices):
        with open(f'{RESULTS_DIR}/i={idx}/shortest_distances.pkl', 'rb') as f:
            shortest_distances = pickle.load(f)
        x_sd = shortest_distances.unique().int()
        with open(f'{RESULTS_DIR}/i={idx}/jac-norms/P={P}/trained.pkl', 'rb') as f:
            jac_norms = pickle.load(f)
        y_sd = bin_jac_norms(jac_norms, shortest_distances, x_sd, agg)
        filter, = torch.where(x_sd<=L)
        binned_jac_norms[i, x_sd[filter]] = y_sd[filter]
    
    for metric_name, ax in zip(metric_names, axs.flatten()):

        mode, error = metric_name.split('_')
        ax.set_title(f'{mode.capitalize()} {error.upper()}')
        ax.set_xlabel('Shortest Distance')
        ax.set_ylabel('Correlation with Mean Sensitivity')

        with open(f'{RESULTS_DIR}/{metric_name}.pkl', 'rb') as f:
            metrics: torch.Tensor = pickle.load(f)

        x = np.arange(binned_jac_norms.size(1))
        y, markers = list(), list()
        for tensor1 in binned_jac_norms.transpose(0, 1):
            tensor1, tensor2 = map(lambda tensor: tensor[~tensor1.isnan()], (tensor1, metrics[P]))
            corr = spearmanr(tensor1, tensor2)
            y.append(corr.statistic)
            markers.append(corr.pvalue<0.1)

        p = ax.plot(x, y, label=f'P = {P}')
        y, markers = map(lambda x: np.array(x), (y, markers))
        ax.scatter(x[markers], y[markers])
        # ax.scatter(x[~markers], y[~markers], facecolors='none', edgecolors=p[-1].get_color())

for ax in axs.flatten():
    ax.grid()
    ax.legend()

fig.tight_layout()
plt.show()