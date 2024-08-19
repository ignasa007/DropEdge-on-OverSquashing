import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth_plot(x, y=None, ax=None, label='', halflife=10):
    
    y_int = y if y is not None else x
    
    x_ewm = pd.Series(y_int).ewm(halflife=halflife)
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    if y is None:
        ax.plot(x_ewm.mean(), label=label, color=color)
        # ax.fill_between(np.arange(x_ewm.mean().shape[0]), x_ewm.mean() + x_ewm.std(), x_ewm.mean() - x_ewm.std(), color=color, alpha=0.15)
    else:
        ax.plot(x, x_ewm.mean(), label=label, color=color)
        # ax.fill_between(x, y_int + x_ewm.std(), y_int - x_ewm.std(), color=color, alpha=0.15)


def main(args):

    args.dataset = args.dataset.split('_')[0]

    with open(f"./results/signal-propagation/{args.vs}/{args.dataset}.pkl", 'rb') as f:
        pairs = pickle.load(f)

    Ps = np.arange(0.0, 1.0, 0.1)
    
    fig, ax = plt.subplots(1, 1)
    
    for P in Ps[::8]:
        data = np.array(pairs[P]).T
        data = data[:, data[0].argsort()]
        data = (data - data.min(axis=1, keepdims=True)) / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True))
        smooth_plot(*data, ax=ax, label=f'P={P:.1f}', halflife=args.halflife)

    ax.set_xlabel(args.vs, fontsize=14)
    ax.set_ylabel('Signal Propagation', fontsize=14)
    ax.set_title(args.dataset, fontsize=16)
    ax.grid()
    ax.legend()

    fig.tight_layout()

    fn = f'assets/signal-propagation/{args.vs}/{args.dataset}.png'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['Proteins', 'MUTAG', 'PTC_MR'])
    parser.add_argument('--vs', type=str, required=True, choices=['Total Resistance', 'Commute Time'])
    parser.add_argument('--halflife', type=int, default=20)
    args = parser.parse_args()
    
    main(args)