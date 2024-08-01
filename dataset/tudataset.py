from typing import Dict
from torch import device as Device, no_grad
from torch_geometric.datasets import TUDataset as TUDatasetTorch
from torch.optim import Optimizer

from dataset.constants import root, Splits, batch_size
from dataset.base import BaseDataset
from dataset.utils import split_dataset, normalize_features, create_loaders
from model import Model


class TUDataset(BaseDataset):

    def __init__(self, name: str, task_name: str, device: Device):

        dataset = TUDatasetTorch(root=root, name=name, use_node_attr=True).to(device)
        dataset = dataset.shuffle()

        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            normalize_features(split_dataset(dataset, Splits.train_split, Splits.val_split, Splits.test_split)),
            batch_size=batch_size,
            shuffle=True
        )

        self.valid_tasks = {'graph-c', }
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(TUDataset, self).__init__(task_name)

    def train(self, model: Model, optimizer: Optimizer):

        model.train()

        for batch in self.train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            train_loss = self.compute_loss(out, batch.y)
            train_loss.backward()
            optimizer.step()

        train_metrics = self.compute_metrics()
        return train_metrics
    
    @no_grad()
    def eval(self, model: Model) -> Dict[str, float]:

        model.eval()
        
        for batch in self.val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.compute_loss(out, batch.y)
        val_metrics = self.compute_metrics()

        for batch in self.test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.compute_loss(out, batch.y)
        test_metrics = self.compute_metrics()

        return val_metrics, test_metrics
    

class Proteins(TUDataset):
    def __init__(self, task_name: str, device: Device):
        super(Proteins, self).__init__(name='PROTEINS', task_name=task_name, device=device)

class PTC(TUDataset):
    def __init__(self, task_name: str, device: Device):
        super(PTC, self).__init__(name='PTC_MR', task_name=task_name, device=device)

class MUTAG(TUDataset):
    def __init__(self, task_name: str, device: Device):
        super(MUTAG, self).__init__(name='MUTAG', task_name=task_name, device=device)