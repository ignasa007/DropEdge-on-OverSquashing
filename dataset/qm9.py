from numpy import mean
from torch import cat, device as Device, cosine_similarity
from torch_geometric.datasets import QM9 as QM9Torch
from torch_geometric.transforms import NormalizeFeatures
from torch.optim import Optimizer

from dataset.constants import root, batch_size, Splits
from model import Model

from typing import Tuple, Iterable
from metrics import Regression


def validate_task(task_name: str, valid_tasks: Iterable, class_name: str = None):

    formatted_name = task_name.replace('_', '-').lower()
    if formatted_name not in valid_tasks:
        raise ValueError('Parameter `task_name` not recognised for the given dataset' +
            + ' ' + f'(got task `{task_name}` for dataset {class_name}).')
    
def set_metrics(task_name: str, num_classes: int) -> Tuple[Regression, int]:

    formatted_name = task_name.replace('_', '-').lower()
    
    output_dim = num_classes  # must be set in child classes
    if formatted_name.endswith('-r'):
        metrics = Regression(num_classes)
    else:
        raise ValueError('Parameter `task_name` not identified.' +
            ' ' + f'Expected `regression`, but got `{task_name}`.')
    
    return metrics, output_dim


class QM9:

    def __init__(self, task_name: str, device: Device):

        dataset = QM9Torch(root=f'{root}/QM9', transform=NormalizeFeatures()).to(device)
        dataset = dataset.shuffle()

        train_end = int(Splits.train_split*len(dataset))
        
        self.train_loader = dataset[:train_end].to_datapipe().batch_graphs(batch_size=batch_size)

        self.valid_tasks = {'graph-r', }
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        
        validate_task(task_name, valid_tasks=self.valid_tasks, class_name=self.__class__.__name__)
        self.metrics, self.output_dim = set_metrics(task_name, num_classes=self.num_classes)

    def train(self, model: Model, optimizer: Optimizer):

        model.train()
        for batch in self.train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            train_loss = self.metrics.update_mse(out, batch.y)
            train_loss.backward()
            optimizer.step()
    
    def eval(self, model: Model, optimizer: Optimizer):

        model.eval()
        for batch in self.train_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.metrics.update_mse(out, batch.y)
        train_loss = self.metrics.compute_mse()
        train_loss.backward()
        full_gradient = cat([param.grad.view(-1) for param in model.parameters()])

        model.train()
        cos_sim = list()
        for batch in self.train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            train_loss = self.metrics.update_mse(out, batch.y)
            train_loss.backward()
            mb_grad_sample = cat([param.grad.view(-1) for param in model.parameters()])
            cos_sim.append(cosine_similarity(mb_grad_sample, full_gradient))

        return mean(cos_sim)