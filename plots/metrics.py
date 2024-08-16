import os
from collections import defaultdict
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'Proteins', 'MUTAG', 'PTC'])
parser.add_argument('--metric', type=str, required=True, choices=['Cross Entropy Loss', 'Accuracy', 'F1 Score'])
parser.add_argument('--gnns', nargs='+', default=('GCN',))
parser.add_argument('--which', type=str, default=['Best'], choices=['Best', 'Final'])
parser.add_argument('--min_depth', type=int, default=2)
parser.add_argument('--max_depth', type=int, default=8)
args = parser.parse_args()

depths = range(args.min_depth, args.max_depth+1)
ncol = np.ceil(len(depths)/1)
ps = np.round(np.arange(0.1, 1, 0.1), decimals=1)

results_dir = f'./results/drop-edge/{args.dataset}'
exp_dir = results_dir + '/{gnn}/L={depth}/P={p}'
# results_dir = f'./results/sensitivity/model-store/{args.dataset}'
# exp_dir = results_dir + '/P={p}'
assets_dir = results_dir.replace('results', 'assets')

### RETRIEVE METRICS ###

train_metrics = defaultdict(list)
test_metrics = defaultdict(list)
gap_metrics = defaultdict(list)

for gnn in args.gnns:
    for depth in tqdm(depths):
        for p in ps:
            exp_dir_format = exp_dir.format(gnn=gnn, depth=depth, p=p)
            for sample_dir in os.listdir(exp_dir_format):
                train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
                if max(train[args.metric]) < 0.5:
                    continue
                if args.which == 'Best':
                    train_metrics[(gnn, depth, p)].append(max(train[args.metric]))
                    test_metrics[(gnn, depth, p)].append(test[args.metric][np.argmax(val[args.metric])])
                elif args.which == 'Final':
                    train_metrics[(gnn, depth, p)].append(train[args.metric][-1])
                    test_metrics[(gnn, depth, p)].append(test[args.metric][-1])
                gap_metrics[(gnn, depth, p)].append(train_metrics[(gnn, depth, p)][-1]-test_metrics[(gnn, depth, p)][-1])

train_metrics = {exp: (np.mean(samples), np.std(samples)) for exp, samples in train_metrics.items()}
test_metrics = {exp: (np.mean(samples), np.std(samples)) for exp, samples in test_metrics.items()}
gap_metrics = {exp: (np.mean(samples), np.std(samples)) for exp, samples in gap_metrics.items()}

### PLOT FOR TRAIN SET ###

fig, axs = plt.subplots(1, len(args.gnns), figsize=(6*len(args.gnns), 4))
if not hasattr(axs, '__len__'): axs = (axs,)

for gnn, ax in zip(args.gnns, axs):
    for depth in depths:
        means, lower, upper = list(), list(), list()
        for drop_p in ps:
            mean, std = train_metrics.get((gnn, depth, drop_p), (np.nan, np.nan))
            means.append(mean); lower.append(mean-std); upper.append(mean+std)
        ax.plot(ps, means, label=f'L = {depth}')
        # ax.fill_between(ps, lower, upper, alpha=0.2)
    ax.set_xlabel('DropEdge Probability', fontsize=12)
    ax.set_ylabel(f'{args.which} Training {args.metric}', fontsize=12)
    ax.set_title(gnn, fontsize=14)
    ax.grid()

fig.suptitle(args.dataset, fontsize=16)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=ncol, bbox_to_anchor = (0, -0.1, 1, 1))
fig.tight_layout()
fn = f'./{assets_dir}/{args.which}/{args.metric}/train.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')

### PLOT FOR TEST SET ###

fig, axs = plt.subplots(1, len(args.gnns), figsize=(6*len(args.gnns), 4))
if not hasattr(axs, '__len__'): axs = (axs,)

for gnn, ax in zip(args.gnns, axs):
    for depth in depths:
        means, lower, upper = list(), list(), list()
        for drop_p in ps:
            mean, std = test_metrics.get((gnn, depth, drop_p), (np.nan, np.nan))
            means.append(mean); lower.append(mean-std); upper.append(mean+std)
        ax.plot(ps, means, label=f'L = {depth}')
        # ax.fill_between(ps, lower, upper, alpha=0.2)
    ax.set_xlabel('DropEdge Probability', fontsize=12)
    ax.set_ylabel(f'{args.which} Training {args.metric}', fontsize=12)
    ax.set_title(gnn, fontsize=14)
    ax.grid()

fig.suptitle(args.dataset, fontsize=16)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=ncol, bbox_to_anchor = (0, -0.1, 1, 1))
fig.tight_layout()
fn = f'./{assets_dir}/{args.which}/{args.metric}/test.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')

### PLOT FOR GEN GAP ###

fig, axs = plt.subplots(1, len(args.gnns), figsize=(6*len(args.gnns), 4))
if not hasattr(axs, '__len__'): axs = (axs,)

for gnn, ax in zip(args.gnns, axs):
    for depth in depths:
        means, lower, upper = list(), list(), list()
        for drop_p in ps:
            mean, std = gap_metrics.get((gnn, depth, drop_p), (np.nan, np.nan))
            means.append(mean); lower.append(mean-std); upper.append(mean+std)
        ax.plot(ps, means, label=f'L = {depth}')
        # ax.fill_between(ps, lower, upper, alpha=0.2)
    ax.set_xlabel('DropEdge Probability', fontsize=12)
    ax.set_ylabel(f'{args.which} Training {args.metric}', fontsize=12)
    ax.set_title(gnn, fontsize=14)
    ax.grid()

fig.suptitle(args.dataset, fontsize=16)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=ncol, bbox_to_anchor = (0, -0.1, 1, 1))
fig.tight_layout()
fn = f'./{assets_dir}/{args.which}/{args.metric}/gen-gap.png'
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')