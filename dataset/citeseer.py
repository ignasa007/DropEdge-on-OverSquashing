from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from dataset.constants import root


def citeseer():

    dataset = Planetoid(root=root, name='CiteSeer', transform=NormalizeFeatures())

    return dataset[0]