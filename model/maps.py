from torch.nn import Identity, ReLU, ELU, Sigmoid, Tanh
from .gcn import GCNLayer


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
        raise ValueError(f'Activation name not recognised (got `{activation_name}`).')
    
    activation_class = activation_map.get(formatted_name)
    
    return activation_class

def get_layer(layer_name: str):

    layer_map = {
        'gcn': GCNLayer,
    }
    
    formatted_name = layer_name.lower()
    if formatted_name not in layer_map:
        raise ValueError(f'GNN layer name not recognised (got `{layer_name}`).')
    
    layer_class = layer_map.get(formatted_name)
    
    return layer_class