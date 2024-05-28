from .node_c import NodeClassification


def verify_task(dataset_name, task_name):

    valid_tasks = {
        'cora': ['node-c', ],
        'citeseer': ['node-c', ],
        'pubmed': ['node-c', ],
    }

    formatted_name = dataset_name.lower()
    if formatted_name not in valid_tasks:
        raise ValueError(f'Dataset argument not recognized (got `{dataset_name}`).')

    formatted_name = task_name.replace('_', '-').lower()
    if formatted_name not in valid_tasks.get(dataset_name):
        raise ValueError(f'Invalid task option passed (got `{task_name}) for dataset {dataset_name}.')
    

def get_head(task_name):

    head_map = {
        'node-c': NodeClassification,
    }

    formatted_name = task_name.replace('_', '-').lower()
    if formatted_name not in head_map:
        raise ValueError(f'Task name not recognised (got `{task_name}`).')
    
    model_head = head_map.get(formatted_name)
    
    return model_head