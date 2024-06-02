from typing import Tuple, Dict
from torch import no_grad, device as Device
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.optim import Optimizer

from dataset.constants import root
from dataset.base import BaseDataset
from model import Model


class Cora(BaseDataset):

    def __init__(self, task_name: str, device: Device):

        dataset = Planetoid(root=root, name='Cora', transform=NormalizeFeatures()).to(device)
        
        self.x = dataset.x
        self.edge_index = dataset.edge_index
        self.y = dataset.y
        
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask

        self.valid_tasks = {'node-c', }
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(Cora, self).__init__(task_name)

    def train(self, model: Model, optimizer: Optimizer) -> Dict[str, float]:

        model.train()
        
        optimizer.zero_grad()
        out = model(self.x, self.edge_index, self.train_mask)
        train_loss = self.compute_loss(out, self.y[self.train_mask])
        train_loss.backward()
        optimizer.step()

        train_metrics = self.compute_metrics()
        return train_metrics
    
    @no_grad()
    def eval(self, model: Model) -> Tuple[Dict[str, float], Dict[str, float]]:

        model.eval()
        out = model(self.x, self.edge_index, mask=None)
        
        self.compute_loss(out[self.val_mask], self.y[self.val_mask])
        val_metrics = self.compute_metrics()
        self.compute_loss(out[self.test_mask], self.y[self.test_mask])
        test_metrics = self.compute_metrics()

        return val_metrics, test_metrics