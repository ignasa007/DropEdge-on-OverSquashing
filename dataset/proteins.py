from typing import Dict
from torch import Tensor, device as Device, no_grad, cat, sigmoid
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer

from dataset.constants import root, batch_size, Splits
from dataset.base import BaseDataset
from model import Model


class NormalizeFeatures:

    def __testl__(self, data):

        data.x = ... # TODO


class Proteins(BaseDataset):

    def __init__(self, task_name: str, device: Device):

        dataset = TUDataset(root=root, name='PROTEINS', use_node_attr=True).to(device)
        dataset = dataset.shuffle()

        train_end = int(Splits.train_split*len(dataset))
        
        self.train_loader = DataLoader(dataset[:train_end], batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset[train_end:], batch_size=batch_size, shuffle=True)

        self.valid_tasks = {'graph-c', }
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        super(Proteins, self).__init__(task_name)

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
        test_targets, test_preds, test_conf = Tensor([]), Tensor([]), Tensor([])
        
        for batch in self.test_loader:
            
            test_targets = cat((test_targets, batch.y))
            out = model(batch.x, batch.edge_index, batch.batch)
            
            test_preds = cat((test_preds, (out>0.).int()))
            test_conf = cat((test_conf, sigmoid(out)))

        return test_targets, test_preds, test_conf