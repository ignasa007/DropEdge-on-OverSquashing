'''
__init__ file for dropout classes.
imports all the dropout classes and creates a function to map the 
    dropout name to the dropout class.
'''

from .dropout import Dropout
from .drop_node import DropNode
from .drop_edge import DropEdge
from .drop_message import DropMessage


map = {
    'dropout': Dropout,
    'drop-node': DropNode,
    'drop-edge': DropEdge,
    'drop-message': DropMessage,
}


def dropout_map(dropout_name):

    '''
    Function to map dropout name to dropout class.

    Args:
        dropout_name (str): name of the dropout used for the experiment
    
    Return:
        dropout_class (BaseDropout): a dropout class if dropout_name is 
            recognized, else None
    '''
    
    dropout_class = map.get(dropout_name.lower(), None)
    
    return dropout_class