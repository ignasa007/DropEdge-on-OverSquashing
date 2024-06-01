from model.ffn.base import BaseHead
from model.ffn.node_r import NodeRegression
from model.ffn.node_c import NodeClassification
from model.ffn.graph_r import GraphRegression
from model.ffn.graph_c import GraphClassification 


def get_head(task_name: str) -> BaseHead:

    head_map = {
        'node-r': NodeRegression,
        'node-c': NodeClassification,
        'graph-r': GraphRegression,
        'graph-c': GraphClassification,
    }

    formatted_name = task_name.replace('_', '-').lower()
    if formatted_name not in head_map:
        raise ValueError(f'Parameter `task_name` not recognised (got `{task_name}`).')
    
    model_head = head_map.get(formatted_name)
    
    return model_head