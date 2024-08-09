import os
from collections import defaultdict
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'Proteins', 'MUTAG', 'PTC'])
parser.add_argument('--metric', type=str, required=True, choices=['Cross Entropy Loss', 'Accuracy', 'F1 Score'])
parser.add_argument('--gnns', nargs='+', required=True)
parser.add_argument('--min_depth', type=int, default=2)
parser.add_argument('--max_depth', type=int, default=8)
args = parser.parse_args()

depths = range(args.min_depth, args.max_depth+1)
ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)

results_dir = f'./results/drop-edge/{args.dataset}'
assets_dir = results_dir.replace('results', 'assets')

train_accs = defaultdict(list)
es_test_accs = defaultdict(list)

for gnn in args.gnns:
    for depth in depths:
        for p in ps:
            exp_dir = f'{results_dir}/{gnn}/L={depth}/P={p}'
            for sample_dir in os.listdir(exp_dir):
                train, val, test = parse_metrics(f'{exp_dir}/{sample_dir}/logs')
                # if max(train[args.metric]) < 0.2:
                #     continue
                train_accs[(gnn, depth, p)].append(max(train[args.metric]))
                es_test_accs[(gnn, depth, p)].append(test[args.metric][np.argmax(val[args.metric])])

train_accs = {exp: (np.mean(samples), np.std(samples)) for exp, samples in train_accs.items()}
es_test_accs = {exp: (np.mean(samples), np.std(samples)) for exp, samples in es_test_accs.items()}

fig, axs = plt.subplots(1, len(args.gnns), figsize=(6*len(args.gnns), 4))
if not hasattr(axs, '__len__'): axs = (axs,)

for gnn, ax in zip(args.gnns, axs):
    for depth in depths:
        lower = ([train_accs[(gnn, depth, drop_p)][0]-train_accs[(gnn, depth, drop_p)][1] for drop_p in ps])
        mean = ([train_accs[(gnn, depth, drop_p)][0] for drop_p in ps])
        upper = ([train_accs[(gnn, depth, drop_p)][0]+train_accs[(gnn, depth, drop_p)][1] for drop_p in ps])
        ax.plot(ps, mean, label=depth)
        ax.fill_between(ps, lower, upper, alpha=0.2)
    ax.set_xlabel('DropEdge Probability')
    ax.set_ylabel(f'Training {args.metric}')
    ax.set_title(gnn)
    ax.legend()
    ax.grid()

fig.tight_layout()
fn = f'./{assets_dir}/train_{args.metric.lower().replace(" ", "-")}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)

fig, axs = plt.subplots(1, len(args.gnns), figsize=(6*len(args.gnns), 4))
if not hasattr(axs, '__len__'): axs = (axs,)

for gnn, ax in zip(args.gnns, axs):
    for depth in depths:
        lower = ([es_test_accs[(gnn, depth, drop_p)][0]-es_test_accs[(gnn, depth, drop_p)][1] for drop_p in ps])
        mean = ([es_test_accs[(gnn, depth, drop_p)][0] for drop_p in ps])
        upper = ([es_test_accs[(gnn, depth, drop_p)][0]+es_test_accs[(gnn, depth, drop_p)][1] for drop_p in ps])
        ax.plot(ps, mean, label=depth)
        ax.fill_between(ps, lower, upper, alpha=0.2)
    ax.set_xlabel('DropEdge Probability')
    ax.set_ylabel(f'ES Test {args.metric}')
    ax.set_title(gnn)
    ax.legend()
    ax.grid()

fig.tight_layout()
fn = f'./{assets_dir}/es-test_{args.metric.lower().replace(" ", "-")}.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)