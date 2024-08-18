import argparse
import os
from glob import glob
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.datasets import TUDataset

from model import Model as Base
from sensitivity.utils import compute_shortest_distances, compute_commute_times


GRAPHS_SAMPLED = 200
NUM_VERTICES_SAMPLED = 10
DROPEDGE_SAMPLES = 10


class Model(Base):
    
    def forward(self, x, edge_index):
    
        for mp_layer in self.message_passing:
            x = mp_layer(x, edge_index)
    
        return x

def initialize_architecture(config):

    model = Model(config)
    for mp_layer in model.message_passing:
        mp_layer.bias = None
    model.train()
    
    return model


def process_graph_datum(datum, config, Ps):

    distances = compute_shortest_distances(datum.edge_index)
    resistance = compute_commute_times(datum.edge_index) / datum.edge_index.size(1)
    
    sources = np.random.choice(datum.num_nodes, min(NUM_VERTICES_SAMPLED, datum.num_nodes), replace=False)
    pairs_runaway = {P: list() for P in Ps}

    for source in sources:

        total_resistance = torch.sum(resistance[source])
        x = torch.zeros_like(datum.x)
        x[source] = torch.randn_like(datum.x[source])
        x[source] = x[source].softmax(dim=-1)
        datum.x.data = x
            
        for P in Ps:

            config.drop_p = P
            model = initialize_architecture(config)

            out = torch.mean(torch.stack([
                model(datum.x, datum.edge_index).detach() for _ in range(DROPEDGE_SAMPLES)
            ]), dim=0)
            out = out.abs()
            # normalize over each feature dimension
            out = (out / out.sum(dim=0, keepdims=True))
            out = torch.nan_to_num(out, nan=0.0)
            propagation_distance = (out*distances[:, [source]]).sum(dim=0).mean() / distances[source].max()

            pairs_runaway[P].append((total_resistance, propagation_distance))

    # averaging over sampled source nodes
    pairs_runaway = {P: tuple(np.mean(pairs_runaway[P], axis=0)) for P in pairs_runaway}
    
    return pairs_runaway


def main(args):

    dataset = TUDataset(root='./data/TUDataset', name=args.dataset, use_node_attr=True).shuffle()
    args.dataset = args.dataset.split('_')[0]
    model_dir = glob(f'./results/drop-edge/{args.dataset}/GCN/L=2/P=0.1/*')[0]
    with open(f'{model_dir}/config.pkl', 'rb') as f:
        config = pickle.load(f)
    config.gnn_layer_sizes = [5] * 10

    Ps = np.arange(0.0, 1.0, 0.1)
    pairs = {P: list() for P in Ps}

    for datum in tqdm(dataset[:GRAPHS_SAMPLED]):
        try:
            arch_runaway = process_graph_datum(datum, config, Ps)
            for P in arch_runaway:
                pairs[P].append(arch_runaway[P])
        except AssertionError:
            continue

    fn = f'./results/signal-propagation/{args.dataset}.pkl'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'wb') as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['Proteins', 'MUTAG', 'PTC_MR'])
    args = parser.parse_args()
    
    main(args)