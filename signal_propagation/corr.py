import argparse
import os
import pickle

import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):

    args.dataset = args.dataset.split('_')[0]

    with open(f'./results/signal-propagation/{args.dataset}.pkl', 'rb') as f:
        pairs = pickle.load(f)

    Ps = np.arange(0.0, 1.0, 0.1)
    data = np.array([list(zip(*pairs[P]))[1] for P in Ps])
    
    corr = wilcoxon(np.expand_dims(data, 0) - np.expand_dims(data, 1), alternative='less', zero_method='zsplit', axis=2)
    
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(corr.statistic, cmap='coolwarm', ax=ax)
    ax.set_title(args.dataset, fontsize=14)
    ax.xaxis.tick_top()
    fig.tight_layout()

    fn = f'assets/signal-propagation/rank-correlation/{args.dataset}.png'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['Proteins', 'MUTAG', 'PTC_MR'])
    args = parser.parse_args()
    
    main(args)