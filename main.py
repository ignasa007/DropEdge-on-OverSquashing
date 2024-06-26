import pickle
from tqdm import tqdm

import torch
from torch.optim import Adam

from dataset import get_dataset, BaseDataset
from model import Model
from utils.config import parse_arguments
from utils.logger import Logger
from utils.format import *


args = parse_arguments()

DEVICE = torch.device(f'cuda:{args.device_index}' if torch.cuda.is_available() and args.device_index is not None else 'cpu')

dataset: BaseDataset = get_dataset(args.dataset, args.task, DEVICE)
model = Model(dataset.num_features, dataset.output_dim, args=args).to(device=DEVICE)
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


logger = Logger(
    dataset=format_dataset_name.get(args.dataset.lower()),
    gnn=format_layer_name.get(args.gnn.lower()),
    dropout=format_dropout_name.get(args.dropout.lower()),
    drop_p=args.drop_p,
)

# TODO: log the new configuration options

logger.log(f'Dataset: {format_dataset_name.get(args.dataset.lower())}', with_time=False)
logger.log(f'Add self-loops: {args.add_self_loops}', with_time=False)
logger.log(f'Normalize edge weights: {args.normalize}\n', with_time=False)

logger.log(f'GNN: {format_layer_name.get(args.gnn.lower())}', with_time=False)
logger.log(f"Number of message-passing steps: {len(args.gnn_layer_sizes)}", with_time=False)
logger.log(f"GNN layers' sizes: {args.gnn_layer_sizes}", with_time=False)
logger.log(f"GNN activation: {args.gnn_activation}\n", with_time=False)

logger.log(f'Task: {format_task_name.get(args.task.lower())}', with_time=False)
logger.log(f"Number of layers in the readout FFN: {len(args.ffn_layer_sizes)+1}", with_time=False)
logger.log(f"FFN layers' sizes: {args.ffn_layer_sizes + [dataset.output_dim]}", with_time=False)
logger.log(f"FFN activation: {args.ffn_activation}\n", with_time=False)

logger.log(f'Dropout: {format_dropout_name.get(args.dropout.lower())}', with_time=False)
logger.log(f'Dropout probability: {args.drop_p}\n', with_time=False)

logger.log(f'Number of training epochs: {args.n_epochs}', with_time=False)
logger.log(f'Learning rate: {args.learning_rate}\n', with_time=False)


for epoch in tqdm(range(1, args.n_epochs+1)):
    logger.log(f'Epoch {epoch}')
    train_metrics = dataset.train(model, optimizer)
    logger.log_metrics(train_metrics, prefix='\tTraining: ', with_time=False, print_text=False)


test_targets, test_preds, test_conf = dataset.eval(model)
torch.save(test_targets, f'{logger.exp_dir}/test-targets.pt')
torch.save(test_preds, f'{logger.exp_dir}/test-predictions.pt')
torch.save(test_conf, f'{logger.exp_dir}/test-confidence.pt')