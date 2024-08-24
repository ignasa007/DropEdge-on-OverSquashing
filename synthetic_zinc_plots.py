import os
from collections import defaultdict
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from utils.parse_logs import parse_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, required=True, choices=['Squared', 'Absolute', 'Absolute Percentage'])
parser.add_argument('--which', type=str, default=['Best'], choices=['Best', 'Final'])
args = parser.parse_args()
args.metric = f'Mean {args.metric} Error'

ps = (0.0, 0.5)
alphas = list(map(lambda x: round(x, 1), np.arange(0.0, 1.01, 0.1)))

assets_dir = './assets/ct-triu/GCN'
exp_dir = assets_dir.replace('assets', 'results') + '/L=11/P={P}/alpha={alpha}'

### RETRIEVE METRICS ###

train_metrics = defaultdict(list)
test_metrics = defaultdict(list)
gap_metrics = defaultdict(list)

for p in ps:
    for alpha in alphas:
        exp_dir_format = exp_dir.format(P=p, alpha=alpha)
        for sample_dir in os.listdir(exp_dir_format):
            train, val, test = parse_metrics(f'{exp_dir_format}/{sample_dir}/logs')
            if len(train[args.metric]) != 250:
                continue
            if args.which == 'Best':
                train_metrics[(p, alpha)].append(min(train[args.metric]))
                test_metrics[(p, alpha)].append(test[args.metric][np.argmin(val[args.metric])])
            elif args.which == 'Final':
                train_metrics[(p, alpha)].append(train[args.metric][-1])
                test_metrics[(p, alpha)].append(test[args.metric][-1])
            gap_metrics[(p, alpha)].append(train_metrics[(p, alpha)][-1]-test_metrics[(p, alpha)][-1])

### PLOT FOR TRAIN SET ###

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
colors = iter(('blue', 'green'))
mean_colors = iter(('purple', 'yellow'))

for p in ps:
    color = next(colors)
    mean_color = next(mean_colors)
    means, lower, upper = list(), list(), list()
    for alpha in alphas:
        metrics = train_metrics.get((p, alpha), (np.nan, np.nan, np.nan))
        ax.scatter([alpha]*len(metrics), metrics, color=color)
        means.append(np.mean(metrics))
        # ax.scatter(alpha, np.mean(metrics), color=mean_color)
    ax.plot(alphas, means, color=color, label=f'P = {p:.1f}')
ax.set_xlabel(r'$\alpha$', fontsize=14)
ax.set_ylabel(f"Training {''.join(map(lambda x: x[0], args.metric.split()))}", fontsize=14)
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(ps), bbox_to_anchor = (0, -0.1, 1, 1))
fig.tight_layout()
fn = f"./{assets_dir}/{args.which}/{args.metric}/train.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')

### PLOT FOR TEST SET ###

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
colors = iter(('blue', 'green'))
mean_colors = iter(('purple', 'yellow'))

for p in ps:
    color = next(colors)
    mean_color = next(mean_colors)
    means, lower, upper = list(), list(), list()
    for alpha in alphas:
        metrics = test_metrics.get((p, alpha), (np.nan, np.nan, np.nan))
        ax.scatter([alpha]*len(metrics), metrics, color=color)
        means.append(np.mean(metrics))
        # ax.scatter(alpha, np.mean(metrics), color=mean_color)
    ax.plot(alphas, means, color=color, label=f'P = {p:.1f}')
ax.set_xlabel(r'$\alpha$', fontsize=14)
ax.set_ylabel(f"Testing {''.join(map(lambda x: x[0], args.metric.split()))}", fontsize=14)
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(ps), bbox_to_anchor = (0, -0.1, 1, 1))
fig.tight_layout()
fn = f"./{assets_dir}/{args.which}/{args.metric}/test.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')

### PLOT FOR GENERALIZATION GAP ###

fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
colors = iter(('blue', 'green'))
mean_colors = iter(('purple', 'yellow'))

for p in ps:
    color = next(colors)
    mean_color = next(mean_colors)
    means, lower, upper = list(), list(), list()
    for alpha in alphas:
        metrics = gap_metrics.get((p, alpha), (np.nan, np.nan, np.nan))
        ax.scatter([alpha]*len(metrics), metrics, color=color)
        means.append(np.mean(metrics))
        # ax.scatter(alpha, np.mean(metrics), color=mean_color)
    ax.plot(alphas, means, color=color, label=f'P = {p:.1f}')
ax.set_xlabel(r'$\alpha$', fontsize=14)
ax.set_ylabel(f"Generalization Gap in {''.join(map(lambda x: x[0], args.metric.split()))}", fontsize=14)
ax.grid()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(ps), bbox_to_anchor = (0, -0.1, 1, 1))
fig.tight_layout()
fn = f"./{assets_dir}/{args.which}/{args.metric}/gen-gap.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn, bbox_inches='tight')