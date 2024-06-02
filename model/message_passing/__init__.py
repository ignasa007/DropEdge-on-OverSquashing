from torch.nn import Identity, ReLU, ELU, Sigmoid, Tanh
from model.message_passing.gcn import GCNLayer
from model.message_passing.gat import GATLayer


def get_activation(activation_name: str):

    activation_map = {
        'identiy': Identity,
        'relu': ReLU,
        'elu': ELU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
    }
    
    formatted_name = activation_name.lower()
    if formatted_name not in activation_map:
        raise ValueError(f'Parameter `activation_name` not recognised (got `{activation_name}`).')
    
    activation_class = activation_map.get(formatted_name)
    
    return activation_class

def get_layer(layer_name: str):

    layer_map = {
        'gcn': GCNLayer,
        'gat': GATLayer,
    }
    
    formatted_name = layer_name.lower()
    if formatted_name not in layer_map:
        raise ValueError(f'Parameter `layer_name` not recognised (got `{layer_name}`).')
    
    layer_class = layer_map.get(formatted_name)
    
    return layer_class