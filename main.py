from tqdm import tqdm

import torch
from torch.optim import Adam

from dataset import get_dataset
from model import Model
from utils.config import parse_arguments
from utils.logger import Logger
from utils.format import *
from utils.results import Results


args = parse_arguments()

DEVICE = torch.device(f'cuda:{args.device_index}' if torch.cuda.is_available() and args.device_index is not None else 'cpu')
# TODO: unify data loaders for node level tasks and graph level tasks
# train_loader, val_loader, test_loader = get_dataset(args.dataset, device=DEVICE)
dataset = get_dataset(args.dataset, args.task).to(device=DEVICE)
model = Model(dataset.num_features, dataset.output_dim, args=args).to(device=DEVICE)
optimizer = Adam(model.parameters(), lr=args.learning_rate)


logger = Logger(
    dataset=args.dataset,
    gnn=args.gnn,
    dropout=args.dropout,
)

logger.log(f'Dataset: {format_dataset_name.get(args.dataset)}', date=False)
logger.log(f'Add self-loops: {args.add_self_loops}', date=False)
logger.log(f'Normalize edge weights: {args.normalize}\n', date=False)

logger.log(f'GNN: {format_layer_name.get(args.gnn)}', date=False)
logger.log(f"Number of message-passing steps: {len(args.gnn_layer_sizes)}", date=False)
logger.log(f"GNN layers' sizes: {args.gnn_layer_sizes}", date=False)
logger.log(f"GNN activation: {args.gnn_activation}\n", date=False)

logger.log(f'Task: {format_task_name.get(args.task)}', date=False)
logger.log(f"Number of layers in the readout FFN: {len(args.ffn_layer_sizes)+1}", date=False)
logger.log(f"FFN layers' sizes: {args.ffn_layer_sizes + [dataset.num_classes]}", date=False)
logger.log(f"FFN activation: {args.ffn_activation}\n", date=False)

logger.log(f'Dropout: {format_dropout_name.get(args.dropout)}', date=False)
logger.log(f'Dropout probability: {args.dropout_prob}\n', date=False)

logger.log(f'Number of training epochs: {args.n_epochs}', date=False)
logger.log(f'Learning rate: {args.learning_rate}\n', date=False)


results = Results()

for epoch in tqdm(range(1, args.n_epochs+1)):

    logger.log(f'\nEpoch {epoch}:')

    model.train()
    for x, edge_index, target, mask in train_loader:
        optimizer.zero_grad()
        loss = model(x, edge_index, target, mask)
        loss.backward()
        optimizer.step()
    logger.log_metrics(model.compute_metrics(), prefix='\tTraining', with_time=False, print_text=True)

    if epoch % args.test_every == 0:
        model.eval()
        with torch.no_grad():
            for x, edge_index, target, mask in val_loader:
                loss = model(x, edge_index, target, mask)
            logger.log_metrics(model.compute_metrics(), prefix='\tValidation', with_time=False, print_text=True)
            for x, edge_index, target, mask in test_loader:
                loss = model(x, edge_index, target, mask)
            logger.log_metrics(model.compute_metrics(), prefix='\tTesting', with_time=False, print_text=True)

    if args.save_every is not None and epoch % args.save_every == 0:
        ckpt_fn = f'{logger.exp_dir}/ckpt-{epoch}.pth'
        logger.log(f'Saving model at {ckpt_fn}.', print_text=True)
        torch.save(model.state_dict(), ckpt_fn)


for dataset in ('training', 'validation', 'testing'):
    logger.save(f'{dataset}_results', results.get(dataset))