from .cora import cora
from .citeseer import citeseer
from .pubmed import pubmed


dataset_map = {
    'cora': cora,
    'citeseer': citeseer,
    'pubmed': pubmed,
}


def get_dataset(dataset_name: str):
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f'Dataset name not recognised (got {dataset_name}).')
    dataset_func = dataset_map.get(dataset_name.lower())
    
    return dataset_func()