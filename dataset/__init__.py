from torch import device as Device
from dataset.base import BaseDataset
from dataset.planetoid import Cora, CiteSeer, PubMed
from dataset.qm9 import QM9
from dataset.tudataset import Proteins, PTC, MUTAG


def get_dataset(dataset_name: str, task_name: str, device: Device) -> BaseDataset:

    dataset_map = {
        'cora': Cora,
        'citeseer': CiteSeer,
        'pubmed': PubMed,
        'qm9': QM9,
        'proteins': Proteins,
        'ptc': PTC,
        'mutag': MUTAG,
    }
    
    formatted_name = dataset_name.lower()
    if formatted_name not in dataset_map:
        raise ValueError(f'Parameter `dataset_name` not recognised (got `{dataset_name}`).')
    
    dataset = dataset_map.get(formatted_name)
    
    return dataset(task_name, device)