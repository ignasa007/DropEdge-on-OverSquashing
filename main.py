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

for epoch in tqdm(range(1, args.n_epochs+1)):
    train_metrics = dataset.train(model, optimizer)