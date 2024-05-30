from dataset.cora import cora
from dataset.citeseer import citeseer
from dataset.pubmed import pubmed


def get_dataset(dataset_name: str):

    dataset_map = {
        'cora': cora,
        'citeseer': citeseer,
        'pubmed': pubmed,
    }
    
    formatted_name = dataset_name.lower()
    if formatted_name not in dataset_map:
        raise ValueError(f'Parameter `dataset_name` not recognised (got `{dataset_name}`).')
    
    dataset = dataset_map.get(formatted_name)
    
    return dataset()