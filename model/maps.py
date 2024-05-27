from torch.nn import Identity, ReLU, ELU, Sigmoid, Tanh
from .gcn import GCNLayer


activation_map = {
    'identiy': Identity,
    'relu': ReLU,
    'elu': ELU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}

layer_map = {
    'gcn': GCNLayer,
}


def get_activation(activation_name: str):
    
    if activation_name.lower() not in activation_map:
        raise ValueError(f'Activation name not recognised (got {activation_name}).')
    activation_class = activation_map.get(activation_name.lower())
    
    return activation_class

def get_layer(layer_name: str):
    
    if layer_name.lower() not in layer_map:
        raise ValueError(f'GNN layer name not recognised (got {layer_name}).')
    layer_class = layer_map.get(layer_name.lower())
    
    return layer_class