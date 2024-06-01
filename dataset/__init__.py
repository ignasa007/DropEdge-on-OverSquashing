from dataset.cora import Cora
from dataset.citeseer import CiteSeer
from dataset.pubmed import PubMed


def get_dataset(dataset_name: str, task_name: str):

    dataset_map = {
        'cora': Cora,
        'citeseer': CiteSeer,
        'pubmed': PubMed,
    }
    
    formatted_name = dataset_name.lower()
    if formatted_name not in dataset_map:
        raise ValueError(f'Parameter `dataset_name` not recognised (got `{dataset_name}`).')
    
    dataset = dataset_map.get(formatted_name)
    
    return dataset(task_name)