from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from .constants import root


def pubmed():

    dataset = Planetoid(root=root, name='PubMed', transform=NormalizeFeatures())
    
    return dataset[0]