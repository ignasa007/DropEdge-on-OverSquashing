from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from dataset.constants import root


def cora():

    dataset = Planetoid(root=root, name='Cora', transform=NormalizeFeatures())
    
    return dataset[0]