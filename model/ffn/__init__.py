from model.ffn.base import BaseHead
from model.ffn.node_c import NodeClassification
 

def get_head(task_name):

    head_map = {
        'node-c': NodeClassification,
    }

    formatted_name = task_name.replace('_', '-').lower()
    if formatted_name not in head_map:
        raise ValueError(f'Parameter `task_name` not recognised (got `{task_name}`).')
    
    model_head = head_map.get(formatted_name)
    
    return model_head