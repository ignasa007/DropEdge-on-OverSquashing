from typing import Iterable

import numpy as np
import torch
from torch_geometric.datasets import QM9 as QM9Torch
from torch_geometric.transforms import NormalizeFeatures

from model import Model


def validate_task(task_name: str, valid_tasks: Iterable, class_name: str = None):

    formatted_name = task_name.replace('_', '-').lower()
    if formatted_name not in valid_tasks:
        raise ValueError('Parameter `task_name` not recognised for the given dataset' +
            + ' ' + f'(got task `{task_name}` for dataset {class_name}).')


class QM9:

    def __init__(self, task_name: str, device: torch.device):

        dataset = QM9Torch(root=f'./data/QM9', pre_transform=NormalizeFeatures()).to(device)
        dataset = dataset.shuffle()[:32_000]
        # taking a small dataset to allow for overfitting
        # (QM9 is too big, won't see the effect of dropout without restricting the size)

        self.train_loader = dataset.to_datapipe().batch_graphs(batch_size=64)
        self.eval_loader = dataset.to_datapipe().batch_graphs(batch_size=64)
        self.train_size = len(dataset)

        self.valid_tasks = {'graph-r', }
        self.num_features = dataset.num_features
        self.output_dim = dataset.num_classes
        
        validate_task(task_name, valid_tasks=self.valid_tasks, class_name=self.__class__.__name__)

        self.batches_trained = 0

    def train(self, model: Model, optimizer: torch.optim.Optimizer):

        model.train()
        means, stdevs = list(), list()
        for batch in self.train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            train_loss = torch.mean(torch.square(out-batch.y))
            train_loss.backward()
            optimizer.step()
            self.batches_trained += 1
            if self.batches_trained % 200 == 0:
                mean, stdev = self.eval(model, optimizer)
                means.append(mean); stdevs.append(stdev)

        return means, stdevs
    
    def eval(self, model: Model, optimizer: torch.optim.Optimizer):

        model.eval()
        optimizer.zero_grad()
        train_loss = torch.Tensor([0.])
        for batch in self.eval_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            train_loss += torch.sum(torch.square(out-batch.y))
        train_loss /= self.train_size
        train_loss.backward()
        full_gradient = torch.cat([param.grad.view(-1) for param in model.parameters()])

        model.train()
        cos_sim = list()
        for i, batch in enumerate(self.eval_loader):
            if i > 100:
                break
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            mb_loss = torch.mean(torch.square(out-batch.y))
            mb_loss.backward()
            mb_grad_sample = torch.cat([param.grad.view(-1) for param in model.parameters()])
            cos_sim.append(torch.cosine_similarity(mb_grad_sample, full_gradient, dim=0))

        return np.mean(cos_sim), np.std(cos_sim)