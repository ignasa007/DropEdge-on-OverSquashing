from dropout.dropout import Dropout
from dropout.drop_node import DropNode
from dropout.drop_edge import DropEdge
from dropout.drop_message import DropMessage
from dropout.drop_gnn import DropGNN
from dropout.drop_agg import DropAgg


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
        raise ValueError(f'Parameter `dropout_name` not recognised (got `{dropout_name}`).')
    
    dropout_class = dropout_map.get(formatted_name)
    
    return dropout_class