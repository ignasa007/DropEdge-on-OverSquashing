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


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str,
    help='The dataset to be trained on [Cora, CiteSeer, PubMed].'
)
parser.add_argument(
    '--task', type=str,
    help='The task to perform with the chosen dataset [Node-C, Graph-C, Graph-R].'
)
parser.add_argument(
    '--gnn_layer', type=str,
    help='The backbone model [GCN, ].'
)
parser.add_argument(
    '--drop_strategy', type=str,
    help='The dropping method [Dropout, Drop-Edge, Drop-Node, Drop-Message, Drop-GNN].'
)
parser.add_argument(
    'opts', default=None, nargs=argparse.REMAINDER, 
    help='Modify any other experiment configurations using the command-line.'
)
args = parser.parse_args()

cfg = Config(
    root='config',
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

logger.log(f'Dataset: {format_dataset_name.get(args.dataset)}', date=False)
logger.log(f'Task: {format_task_name.get(args.task)}', date=False)
logger.log(f'GNN: {format_layer_name.get(args.gnn_layer)}', date=False)
logger.log(f'Dropout: {format_dropout_name.get(args.dropout_strategy)}', date=False)

logger.log(f'Add self-loops: {cfg.add_self_loops}', date=False)
logger.log(f'Normalize edge weights: {cfg.normalize}', date=False)

logger.log(f'Number of layers: {len(cfg.h_layer_sizes)+1}', date=False)
logger.log(f"Layers' sizes: {[dataset.num_features] + cfg.h_layer_sizes + [dataset.num_classes]}", date=False)
logger.log(f'Activation: {format_activation_name.get(cfg.activation)}', date=False)

logger.log(f'Drop probability: {cfg.dropout_prob}\n', date=False)


logger.log(f'Starting training...', print_text=True)
results = Results()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(dataset.x, dataset.edge_index)
    loss = F.cross_entropy(out[dataset.train_mask], dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(dataset.x, dataset.edge_index)
    _, pred = out.max(dim=1)
    train_correct = int(pred[dataset.train_mask].eq(dataset.y[dataset.train_mask]).sum().item())
    train_acc = train_correct / int(dataset.train_mask.sum())
    validate_correct = int(pred[dataset.val_mask].eq(dataset.y[dataset.val_mask]).sum().item())
    validate_acc = validate_correct / int(dataset.val_mask.sum())
    test_correct = int(pred[dataset.test_mask].eq(dataset.y[dataset.test_mask]).sum().item())
    test_acc = test_correct / int(dataset.test_mask.sum())
    return train_acc, validate_acc, test_acc

model.train()
for epoch in tqdm(range(1, cfg.n_epochs+1)):

    

    if isinstance(cfg.save_every, int) and epoch % cfg.save_every == 0:
        ckpt_fn = f'{logger.EXP_DIR}/ckpt-{epoch}.pth'
        logger.log(f'Saving model at {ckpt_fn}.', print_text=True)
        torch.save(model.state_dict(), ckpt_fn)


for dataset in ('training', 'validation', 'testing'):
    logger.save(f'{dataset}_results', results.get(dataset))