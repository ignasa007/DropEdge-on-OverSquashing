from typing import Dict
from torch import device as Device, no_grad
from torch_geometric.datasets import LRGBDataset as LRGBDatasetTorch
from torch.optim import Optimizer

from dataset.constants import root
from dataset.base import BaseDataset
from dataset.utils import normalize_features, create_loaders
from model import Model


batch_size = 20     # TODO: tunable


class LRGBDataset(BaseDataset):

    def __init__(self, name: str, task_name: str, device: Device):

        train, val, test = (
            LRGBDatasetTorch(root=root, name=name, split=split).to(device).shuffle()
            for split in ('train', 'val', 'test')
        )

        self.train_loader, self.val_loader, self.test_loader = create_loaders(
            normalize_features(train, val, test),
            batch_size=batch_size,
            shuffle=True
        )

        self.valid_tasks = {'node-c', }
        self.num_features = train.num_features
        self.num_classes = train.num_classes
        super(LRGBDataset, self).__init__(task_name)

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
    

class Pascal(LRGBDataset):
    def __init__(self, task_name: str, device: Device):
        super(Pascal, self).__init__(name='PascalVOC-SP', task_name=task_name, device=device)