from .dropout import Dropout
from .drop_node import DropNode
from .drop_edge import DropEdge
from .drop_message import DropMessage
from .drop_gnn import DropGNN
from .drop_agg import DropAgg


dropout_map = {
    'dropout': Dropout,
    'drop-node': DropNode, 'dropnode': DropNode,
    'drop-edge': DropEdge, 'dropedge': DropEdge,
    'drop-message': DropMessage, 'dropmessage': DropMessage,
    'drop-gnn': DropGNN, 'dropgnn': DropGNN,
    'drop-agg': DropAgg, 'dropagg': DropAgg,
}


def get_dropout(dropout_name: str):

    if dropout_name.lower() not in dropout_map:
        raise ValueError(f'Dropout name not recognised (got {dropout_name}).')
    dropout_class = dropout_map.get(dropout_name.lower())
    
    return dropout_class