import argparse


def layer_sizes(args):

    out = list()
    for arg in args:
        if isinstance(arg, str) and '*' in arg:
            size, mult = map(int, arg.split('*'))
            out.extend([size]*mult)
        elif isinstance(arg, str) and arg.isdigit() or isinstance(arg, int):
            out.append(int(arg))
        else:
            raise ValueError(f'arg = {arg}, type(arg) = {type(arg)}')

    return out


def parse_arguments(return_unknown=False):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', type=str, required=True, choices=['Cora', 'CiteSeer', 'PubMed', 'Proteins', 'MUTAG', 'PTC', 'QM9', 'ZINC', 'Pascal'],
        help='The dataset to be trained on.'
    )
    parser.add_argument(
        '--add_self_loops', type=bool, default=True,
        help='Boolean value indicating whether to add self-loops during message passing.'
    )
    parser.add_argument(
        '--normalize', type=bool, default=True,
        help='Boolean value indicating whether to normalize edge weights during message passing.'
    )

    parser.add_argument(
        '--gnn', type=str, required=True, choices=['GCN', 'GAT', 'APPNP'],
        help='The backbone model.'
    )
    parser.add_argument(
        '--gnn_layer_sizes', type=str, nargs='+', default=[16, 16],
        help="Hidden layers' sizes for the GNN."
    )
    parser.add_argument(
        '--attention_heads', type=int, default=None,
        help='Number of attention heads (when GNN is GAT).'
    )
    parser.add_argument(
        '--power_iter', type=int, default=None,
        help='Number of power iteration steps (when GNN is APPNP).'
    )
    parser.add_argument(
        '--teleport_p', type=float, default=None,
        help='Teleport probability to use (when GNN is APPNP).'
    )
    parser.add_argument(
        '--gnn_activation', type=str, default='ReLU', choices=['Identity', 'ReLU', 'ELU', 'GeLU', 'Sigmoid', 'Tanh'],
        help='The non-linearity to use for message-passing.'
    )

    parser.add_argument(
        '--task', type=str, required=True, choices=['Node-C', 'Graph-C', 'Graph-R'],
        help='The task to perform with the chosen dataset.'
    )
    parser.add_argument(
        '--ffn_layer_sizes', type=str, nargs='*', default=[],
        help="Hidden layers' sizes for the readout FFN."
    )
    parser.add_argument(
        '--ffn_activation', type=str, default='ReLU', choices=['Identity', 'ReLU', 'ELU', 'GeLU', 'Sigmoid', 'Tanh'],
        help='The non-linearity to use for readout.'
    )

    parser.add_argument(
        '--dropout', type=str, required=True, choices=['Dropout', 'DropEdge', 'DropNode', 'DropMessage', 'DropGNN'],
        help='The dropping method.'
    )
    parser.add_argument(
        '--drop_p', type=float, default=0.5,
        help='The dropping probability used with the dropout method.'
    )

    parser.add_argument(
        '--n_epochs', type=int, default=500,
        help='Number of epochs to train the model for.'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-2,
        help='Learning rate for Adam optimizer.'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=5e-3,
        help='Weight decay for Adam optimizer.'
    )

    parser.add_argument(
        '--device_index', type=int, default=None,
        help="Index of the GPU to use; skip if you're using CPU."
    )
    parser.add_argument(
        '--test_every', type=int, default=1,
        help='Number of epochs of training to test after.\n' \
            '\tSpecial cases: -1 to test only at the last epoch.'
    )
    parser.add_argument(
        '--save_every', type=int, default=None,
        help='Number of epochs of training to save the model after.\n' \
            '\tSpecial cases: skip to never save and -1 to save at the last epoch.'
    )

    config, others = parser.parse_known_args()
    config.gnn_layer_sizes = layer_sizes(config.gnn_layer_sizes)
    config.ffn_layer_sizes = layer_sizes(config.ffn_layer_sizes)

    if not return_unknown:
        return config

    other_args = argparse.Namespace()
    for i in range(0, len(others), 2):
        key = others[i].lstrip('--')
        value = others[i+1]
        setattr(other_args, key, value)
    
    return config, other_args