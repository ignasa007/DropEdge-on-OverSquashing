import os
import pickle

import torch
import matplotlib.pyplot as plt

from sensitivity.utils import bin_jac_norms


sensitivity_dir = '../results/sensitivity/2024-07-27-14-46-51'

max_shortest_distance = 0.
for i_dir in os.listdir(sensitivity_dir):
    if 'copy' in i_dir: continue
    i_dir = f'{sensitivity_dir}/{i_dir}'
    with open (f'{i_dir}/shortest_distances.pkl', 'rb') as f:
        shortest_distances = pickle.load(f)
    max_shortest_distance = max(max_shortest_distance, shortest_distances.max())
max_shortest_distance = int(max_shortest_distance)

Ps = torch.arange(0, 81, 10).numpy()
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for trained, ax in zip(('untrained', 'trained'), axs):
    
    ax.set_title(f'{trained.capitalize()} Models')
    mean_jac_norms = dict()

    for P in Ps:

        sum_jac_norms = torch.zeros(max_shortest_distance+1)
        count_jac_norms = torch.zeros_like(sum_jac_norms)

        for i_dir in os.listdir(sensitivity_dir):

            i_dir = f'{sensitivity_dir}/{i_dir}'
            if not os.path.isdir(i_dir) or 'copy' in i_dir: continue
            
            with open (f'{i_dir}/shortest_distances.pkl', 'rb') as f:
                shortest_distances = pickle.load(f)
            x_sd = shortest_distances.unique().int()

            with open(f'{i_dir}/jac-norms/{trained}/p={P}.pkl', 'rb') as f:
                jac_norms = pickle.load(f)
            y_sd = bin_jac_norms(jac_norms, shortest_distances, x_sd)
            
            sum_jac_norms[x_sd] += y_sd
            count_jac_norms[x_sd] += 1

        mean_jac_norms[P] = sum_jac_norms / count_jac_norms
        
    for k, v in mean_jac_norms.items():
        if k != 0.:
            delta = 100 * (v / mean_jac_norms[0] - 1.)
            delta[delta>1000.] = torch.nan
            ax.plot(torch.arange(v.size(0)), delta, label=f'P = {k/100:.1f}')

    ax.set_xlabel('Shortest Distances')
    ax.set_ylabel('Sensitivity Change (%)')
    # ax.set_yscale('log')
    ax.grid()
    ax.legend()

fig.tight_layout()
plt.show()