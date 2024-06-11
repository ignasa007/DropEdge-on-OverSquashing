from os import makedirs
from pickle import dump, HIGHEST_PROTOCOL
from tqdm import tqdm

import torch
from torch.optim import Adam

from dataset import get_dataset
from model import Model
from utils.config import parse_arguments


args = parse_arguments()

DEVICE = torch.device(f'cuda:{args.device_index}' if torch.cuda.is_available() and args.device_index is not None else 'cpu')

dataset = get_dataset('QM9', 'Graph-R', DEVICE)
model = Model(dataset.num_features, dataset.output_dim, args=args).to(device=DEVICE)
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


grad_norms = list()
for epoch in tqdm(range(1, args.n_epochs+1)):
    grad_norms.extend(dataset.train(model, optimizer))

EXP_DIR = f'./results/grad-norm/{args.dataset}/{args.gnn}/{args.dropout}/prob={int(100*args.drop_p)}'
makedirs(EXP_DIR)

with open(f'{EXP_DIR}/grad-norms.pkl', 'wb') as f:
    dump(grad_norms, f, protocol=HIGHEST_PROTOCOL)