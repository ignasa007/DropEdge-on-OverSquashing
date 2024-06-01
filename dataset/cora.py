from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from dataset.constants import root
from dataset.base import BaseDataset
from model import Model


class Cora(BaseDataset):

    def __init__(self, task_name: str):

        self.valid_tasks = {'node-c', }
        super(Cora, self).__init__(task_name)

        dataset = Planetoid(root=root, name='Cora', transform=NormalizeFeatures())
        
        self.x = dataset.x
        self.edge_index = dataset.edge_index
        self.y = dataset.y
        
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask

    def forward(self, model: Model):

        if model.training:
            loss = model(self.x, self.edge_index, self.y, self.train_mask, return_preds=True)
            return loss
        else:
            


train_loader = [(dataset.x, dataset.edge_index, dataset.y, dataset.train_mask)]
val_loader   = [(dataset.x, dataset.edge_index, dataset.y, dataset.val_mask)]
test_loader  = [(dataset.x, dataset.edge_index, dataset.y, dataset.test_mask)]

# TODO: return only predictions from the model, not loss
# TODO: metrics under dataset instead of model head