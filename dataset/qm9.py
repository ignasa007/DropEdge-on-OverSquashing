from typing import Tuple, Dict
from torch import no_grad, device as Device
from torch_geometric.datasets import QM9 as QM9Torch
from torch_geometric.transforms import NormalizeFeatures
from torch.optim import Optimizer

from dataset.constants import root, batch_size, Splits
from dataset.base import BaseDataset
from model import Model


class QM9(BaseDataset):

    def __init__(self, task_name: str, device: Device):

        dataset = QM9Torch(root=root, transform=NormalizeFeatures()).to(device)
        dataset = dataset.shuffle()

        train_end = int(Splits.train_split*len(dataset))
        val_end = train_end + int(Splits.val_split*len(dataset))
        
        # batch_graphs takes an additional named argument $drop_last (bool):
        #       whether to drop the last batch (possibly smaller than $batch_size) or not
        self.train_loader = dataset[:train_end].to_datapipe().batch_graphs(batch_size=batch_size)
        self.val_loader = dataset[train_end:val_end].to_datapipe().batch_graphs(batch_size=batch_size)
        self.test_loader = dataset[val_end:].to_datapipe().batch_graphs(batch_size=batch_size)

        self.valid_tasks = {'graph-r', }
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(QM9, self).__init__(task_name)

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
    def eval(self, model: Model) -> Tuple[Dict[str, float], Dict[str, float]]:

        model.eval()
        
        for batch in self.train_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.compute_loss(out, batch.y)
        val_metrics = self.compute_metrics()

        for batch in self.train_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            self.compute_loss(out, batch.y)
        test_metrics = self.compute_metrics()

        return val_metrics, test_metrics