import os
from glob import glob
import pickle

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader.dataloader import Collater

from utils.parse_logs import parse_configs
from utils.format import format_dataset_name
from model import Model
from metrics import Classification


DATASET = 'PROTEINS'
L = 7
MODEL_DIR = f'./results/model-store/{format_dataset_name[DATASET.lower()]}/L={L}'
MODEL_SAMPLES = 10
RESULTS_DIR = f'./results/sensitivity/{format_dataset_name[DATASET.lower()]}/L={L}'

dataset = TUDataset(root='./data', name=DATASET, use_node_attr=True)
indices = [dir_name for dir_name in os.listdir(RESULTS_DIR) if dir_name[-1].isdigit()]
indices = list(map(lambda x: int(x.split('=')[1]), indices))

input = Collater(dataset)(dataset[indices])
mask = input.batch
target = input.y

dir_names = {float(parse_configs(fn)['drop_p']): os.path.dirname(fn) for fn in glob(f'{MODEL_DIR}/*/logs')}
metric = Classification(dataset.num_classes)
nonlinearity = metric.nonlinearity
ce_loss = metric.loss_fn; ce_loss.reduction = 'none'
if dataset.num_classes == 2:
    mae_loss = lambda preds, target: torch.abs(preds - target)
else:
    mae_loss = lambda preds, target: torch.abs(1 - preds[:, target])

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
    logits = torch.zeros(len(indices), config.output_dim)
    for _ in range(n_samples):
        logits += model(input.x, input.edge_index, mask).detach().squeeze()
    logits /= n_samples
    train_ce[P] = ce_loss(logits, target)
    train_mae[P] = mae_loss(nonlinearity(logits), target)

    model.eval()
    logits = model(input.x, input.edge_index, mask).detach().squeeze()
    eval_ce[P] = ce_loss(logits, target)
    eval_mae[P] = mae_loss(nonlinearity(logits), target)

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