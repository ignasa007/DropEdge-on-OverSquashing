import argparse
from tqdm import tqdm

import torch
from torch.optim import Adam

from dataset import get_dataset
from model import Model
from model.heads import verify_task
from utils.config import Config
from utils.logger import Logger
from utils.format import *
from utils.results import Results


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, required=True,
    help='The dataset to be trained on [Cora, CiteSeer, PubMed].'
)
parser.add_argument(
    '--task', type=str, required=True,
    help='The task to perform with the chosen dataset [Node-C, Graph-C, Graph-R].'
)
parser.add_argument(
    '--gnn_layer', type=str, required=True,
    help='The backbone model [GCN, ].'
)
parser.add_argument(
    '--drop_strategy', type=str, required=True,
    help='The dropping method [Dropout, Drop-Edge, Drop-Node, Drop-Message, Drop-GNN].'
)
parser.add_argument(
    'opts', default=None, nargs=argparse.REMAINDER,
    help='Modify any other experiment configurations using the command-line.'
)
args = parser.parse_args()

verify_task(args.dataset, args.task)

cfg = Config(
    root='config',
    override=args.opts,
)


DEVICE = torch.device(f'cuda:{cfg.device_index}' if torch.cuda.is_available() and cfg.device_index is not None else 'cpu')
# TODO: unify data loaders for 
#   1. node level tasks -- one graph + split masks
#   2. graph level tasks -- several graphs in each split
# REMINDER: num_classes = 2 => model head has a single output; loss is binary cross-entropy
#           num_classes > 2 => model has $num_classes outputs; loss is cross-entropy
dataset = get_dataset(args.dataset).to(device=DEVICE)
# train_loader, val_loader, test_loader = get_dataset(args.dataset, device=DEVICE)
model = Model(
    input_dim=dataset.num_features,
    h_layer_sizes=cfg.h_layer_sizes,
    output_dim=dataset.num_classes, # TODO: what is this for regression tasks?
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

logger.log(f'Dataset: {format_dataset_name.get(args.dataset)}', date=False)
logger.log(f'Task: {format_task_name.get(args.task)}', date=False)
logger.log(f'GNN: {format_layer_name.get(args.gnn_layer)}', date=False)
logger.log(f'Dropout: {format_dropout_name.get(args.dropout_strategy)}', date=False)

logger.log(f'Add self-loops: {cfg.add_self_loops}', date=False)
logger.log(f'Normalize edge weights: {cfg.normalize}', date=False)

logger.log(f'Number of layers: {len(cfg.h_layer_sizes)+1}', date=False)
logger.log(f"Layers' sizes: {[dataset.num_features] + cfg.h_layer_sizes + [dataset.num_classes]}", date=False)
logger.log(f'Activation: {format_activation_name.get(cfg.activation)}', date=False)

logger.log(f'Dropout probability: {cfg.dropout_prob}', date=False)


results = Results()

for epoch in tqdm(range(1, cfg.n_epochs+1)):

    logger.log(f'\nEpoch {epoch}:')

    model.train()
    for inputs, target, mask in train_loader:
        optimizer.zero_grad()
        loss, metrics = model(*inputs, target, mask)
        loss.backward()
        optimizer.step()
        logger.log_metrics(metrics, prefix='Training', with_time=False, print_text=True)

    if epoch % cfg.test_every == 0:
        model.eval()
        with torch.no_grad():
            for inputs, target, mask in val_loader:
                loss, metrics = model(*inputs, target, mask)
                logger.log_metrics(metrics, prefix='Validation', with_time=False, print_text=True)
            for inputs, target, mask in test_loader:
                loss, metrics = model(*inputs, target, mask)
                logger.log_metrics(metrics, prefix='Testing', with_time=False, print_text=True)

    if isinstance(cfg.save_every, int) and epoch % cfg.save_every == 0:
        ckpt_fn = f'{logger.EXP_DIR}/ckpt-{epoch}.pth'
        logger.log(f'Saving model at {ckpt_fn}.', print_text=True)
        torch.save(model.state_dict(), ckpt_fn)


for dataset in ('training', 'validation', 'testing'):
    logger.save(f'{dataset}_results', results.get(dataset))