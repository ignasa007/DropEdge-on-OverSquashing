from .dropout import Dropout
from .drop_node import DropNode
from .drop_edge import DropEdge
from .drop_message import DropMessage
from .drop_gnn import DropGNN
from .drop_agg import DropAgg


def get_dropout(dropout_name: str):

    dropout_map = {
        'dropout': Dropout,
        'dropnode': DropNode,
        'dropedge': DropEdge,
        'dropmessage': DropMessage,
        'dropgnn': DropGNN,
        'dropagg': DropAgg,
    }

    formatted_name = dropout_name.replace('-', '').lower()
    if formatted_name not in dropout_map:
        raise ValueError(f'Dropout name not recognised (got `{dropout_name}`).')
    
    dropout_class = dropout_map.get(formatted_name)
    
    return dropout_class