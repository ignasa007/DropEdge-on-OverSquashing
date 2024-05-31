import argparse


def validate_task(dataset_name, task_name):

    valid_tasks = {
        'cora': ['node-c', ],
        'citeseer': ['node-c', ],
        'pubmed': ['node-c', ],
    }

    formatted_name = dataset_name.lower()
    if formatted_name not in valid_tasks:
        raise ValueError(f'Parameter `dataset_name` not recognized (got `{dataset_name}`).')

    formatted_name = task_name.replace('_', '-').lower()
    if formatted_name not in valid_tasks.get(dataset_name):
        raise ValueError('Parameter `task_name` not recognised for the passed `dataset_name`' \
            f'(got task `{task_name}` for dataset `{dataset_name}`).')


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', type=str, required=True,
        help='The dataset to be trained on: [Cora, CiteSeer, PubMed].'
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
        '--gnn', type=str, required=True,
        help='The backbone model: [GCN, ].'
    )
    parser.add_argument(
        '--gnn_layer_sizes', type=list, default=[16, 16],
        help="Hidden layers' sizes for the GNN."
    )
    parser.add_argument(
        '--gnn_activation', type=str, default='ReLU',
        help='The non-linearity to use for message-passing: [Identity, ReLU, ELU, GeLU, Sigmoid, Tanh].'
    )

    parser.add_argument(
        '--task', type=str, required=True,
        help='The task to perform with the chosen dataset: [Node-C, Graph-C, Graph-R].'
    )
    parser.add_argument(
        '--ffn_layer_sizes', type=list, default=[],
        help="Hidden layers' sizes for the readout FFN."
    )
    parser.add_argument(
        '--ffn_activation', type=str, default='ReLU',
        help='The non-linearity to use for readout: [Identity, ReLU, ELU, GeLU, Sigmoid, Tanh].'
    )

    parser.add_argument(
        '--dropout', type=str, required=True,
        help='The dropping method [Dropout, Drop-Edge, Drop-Node, Drop-Message, Drop-GNN].'
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
        '--learning_rate', type=float, default=5e-3,
        help='Learning rate for Adam optimizer.'
    )

    parser.add_argument(
        '--device_index', type=int, default=None,
        help="Index of the GPU to use; skip if you're using CPU."
    )
    parser.add_argument(
        '--test_every', type=int, default=1,
        help='Number of epochs of training to test after.'
    )
    parser.add_argument(
        '--save_every', type=int, default=None,
        help='Number of epochs of training to save the model after.'
    )

    args = parser.parse_args()

    assert validate_task(args.dataset, args.task)

    return args