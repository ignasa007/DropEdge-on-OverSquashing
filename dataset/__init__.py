from torch import device as Device
from dataset.base import BaseDataset
from dataset.cora import Cora
from dataset.qm9 import QM9


def get_dataset(dataset_name: str, task_name: str, device: Device) -> BaseDataset:

    dataset_map = {
        'cora': Cora,
        'qm9': QM9,
    }
    
    formatted_name = dataset_name.lower()
    if formatted_name not in dataset_map:
        raise ValueError(f'Parameter `dataset_name` not recognised (got `{dataset_name}`).')
    
    dataset = dataset_map.get(formatted_name)
    
    return dataset(task_name, device)