import os
from glob import glob
import pickle
import argparse

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch_geometric.datasets import Planetoid

from model import Model
from utils.parse_logs import parse_configs
from utils.format import format_dataset_name


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--L', type=int)
args = parser.parse_args()

DATASET = args.dataset
L = args.L
MODEL_DIR = f'./results/model-store/{format_dataset_name[DATASET.lower()]}/L={L}'
MODEL_SAMPLES = 10
RESULTS_DIR = f'./results/sensitivity/{format_dataset_name[DATASET.lower()]}/L={L}'

dataset = Planetoid(root='./data', name=DATASET, split='full')
indices = [dir_name for dir_name in os.listdir(RESULTS_DIR) if dir_name[-1].isdigit()]
indices = list(map(lambda x: int(x.split('=')[1]), indices))

input = dataset
mask = indices
target = input.y[mask]

dir_names = {float(parse_configs(fn)['drop_p']): os.path.dirname(fn) for fn in glob(f'{MODEL_DIR}/*/logs')}
if dataset.num_classes == 2:
    nonlinearity = torch.sigmoid
    ce_loss = lambda logits, target: BCEWithLogitsLoss(reduction='none')(logits, target.float())
    mae_loss = lambda logits, target: torch.abs(nonlinearity(logits) - target)
else:
    nonlinearity = lambda probs: torch.softmax(probs, dim=-1)
    ce_loss = CrossEntropyLoss(reduction='none')
    mae_loss = lambda logits, target: torch.abs(1 - nonlinearity(logits)[:, target])

train_ce, train_mae, eval_ce, eval_mae = dict(), dict(), dict(), dict()

for P, dir_name in dir_names.items():

    with open(f'{dir_name}/config.pkl', 'rb') as f:
        config = pickle.load(f)

    model = Model(config)
    state_dict = torch.load(f'{dir_name}/ckpt-400.pt')
    model.load_state_dict(state_dict)
    
    # averaging logits to improve confidence calibration, since GNNs are usually underconfident
    # https://ceur-ws.org/Vol-3215/19.pdf
    n_samples = MODEL_SAMPLES if P > 0. else 1
    model.train()
    logits = torch.zeros(len(mask), config.output_dim)
    for _ in range(n_samples):
        logits += model(input.x, input.edge_index, mask).detach().squeeze()
    logits /= n_samples
    train_ce[P] = ce_loss(logits, target)
    train_mae[P] = mae_loss(logits, target)

    model.eval()
    logits = model(input.x, input.edge_index, mask).detach().squeeze()
    eval_ce[P] = ce_loss(logits, target)
    eval_mae[P] = mae_loss(logits, target)

with open(f'{RESULTS_DIR}/indices.pkl', 'wb') as f:
    pickle.dump(indices, f, pickle.HIGHEST_PROTOCOL)
with open(f'{RESULTS_DIR}/train_ce.pkl', 'wb') as f:
    pickle.dump(train_ce, f, pickle.HIGHEST_PROTOCOL)
with open(f'{RESULTS_DIR}/train_mae.pkl', 'wb') as f:
    pickle.dump(train_mae, f, pickle.HIGHEST_PROTOCOL)
with open(f'{RESULTS_DIR}/eval_ce.pkl', 'wb') as f:
    pickle.dump(eval_ce, f, pickle.HIGHEST_PROTOCOL)
with open(f'{RESULTS_DIR}/eval_mae.pkl', 'wb') as f:
    pickle.dump(eval_mae, f, pickle.HIGHEST_PROTOCOL)