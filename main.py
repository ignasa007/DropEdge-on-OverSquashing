import argparse
from tqdm import tqdm

import torch
from torch.optim import Adam

from dataset import get_dataset
from model import Model
from utils.config import Config
from utils.logger import Logger
from utils.format import *
from utils.metrics import Results


# TODO: add options for all remaining model parameters

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str,
    help='The dataset to be trained on [Cora, CiteSeer, PubMed].'
)
parser.add_argument(
    '--gnn_layer', type=str,
    help='The backbone model [GCN, ].'
)
parser.add_argument(
    '--drop_strategy', type=str,
    help='The dropping method [Dropout, DropEdge, DropNode, DropMessage, DropGNN].'
)
parser.add_argument(
    'opts', default=None, nargs=argparse.REMAINDER, 
    help='Modify any other experiment configurations using the command-line.'
)
args = parser.parse_args()

cfg = Config(
    root='config',
    dataset=args.dataset,
    gnn_layer=args.gnn_layer,
    drop_strategy=args.drop_strategy,
    override=args.opts,
)


DEVICE = torch.device(f'cuda:{cfg.device_index}' if torch.cuda.is_available() and cfg.device_index is not None else 'cpu')
dataset = get_dataset(args.dataset).to(device=DEVICE)
model = Model(
    input_dim=dataset.num_features,
    h_layer_sizes=cfg.h_layer_sizes,
    output_dim=dataset.num_classes,
    gnn_layer=args.gnn_layer,
    add_self_loops=cfg.add_self_loops,
    normalize=cfg.normalize,
    drop_strategy=args.drop_strategy,
    dropout_prob=cfg.dropout_prob,
    activation=cfg.activation,
).to(device=DEVICE)
optimizer = Adam(model.parameters(), lr=cfg.lr)


logger = Logger(
    dataset=args.dataset,
    gnn_layer=args.gnn_layer,
    drop_strategy=args.drop_strategy,
)

logger.log(f'Dataset: {format_dataset_name(args.dataset)}')
logger.log(f'Add self-loops: {cfg.add_self_loops}')
logger.log(f'Normalize edge weights: {cfg.normalize}')
logger.log(f'GNN type: {format_layer_name(args.gnn_layer)}')
logger.log(f'Number of layers: {len(cfg.h_layer_sizes)+1}')
logger.log(f"Layers' sizes: {[dataset.num_features] + cfg.h_layer_sizes + [dataset.num_classes]}")
logger.log(f'Activation: {format_activation_name(cfg.activation)}')
logger.log(f'Dropout method: {format_dropout_name(args.dropout_strategy)}')
logger.log(f'Dropout probability: {cfg.dropout_prob}\n')
# TODO: add task type -- node classification, graph classification, node regression, ...


logger.log(f'Starting training...', print_text=True)
results = Results()
log_train = log_val = log_test = True
save_model = False


for epoch in tqdm(range(cfg.n_epochs)):

    if save_model:
        ckpt_fn = f'{logger.SAVE_DIR}/ckpt.pth'
        logger.log(f'Saving model at {ckpt_fn}.', print_text=True)
        torch.save(model.state_dict(), ckpt_fn)


for dataset in ('training', 'validation', 'testing'):
    logger.save(f'{dataset}_results', results.get(dataset))