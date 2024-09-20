format_dataset_name = {
    'cora': 'Cora',
    'citeseer': 'CiteSeer',
    'pubmed': 'PubMed',
    'qm9': 'QM9',
    'proteins': 'Proteins',
    'ptc': 'PTC',
    'mutag': 'MUTAG',
    'pascal': 'Pascal',
    'syntheticzinc': 'SyntheticZINC',
    'syntheticmutag': 'SyntheticMUTAG',
    'twitchde': 'TwitchDE',
    'actor': 'Actor',
    'chameleon': 'Chameleon',
    'crocodile': 'Crocodile',
    'squirrel': 'Squirrel',
}

format_task_name = {
    'node-c': 'Node-C', 'node_c': 'Node-C',
    'graph-c': 'Graph-C', 'graph_c': 'Graph-C',
    'graph-r': 'Graph-R', 'graph_r': 'Graph-R',
}

format_layer_name = {
    'gcn': 'GCN',
    'gat': 'GAT',
    'appnp': 'APPNP',
}

format_dropout_name = {
    'dropout': 'Dropout',
    'drop-edge': 'DropEdge', 'dropedge': 'DropEdge',
    'drop-node': 'DropNode', 'dropnode': 'DropNode',
    'drop-message': 'DropMessage', 'dropmessage': 'DropMessage',
    'drop-gnn': 'DropGNN', 'dropgnn': 'DropGNN',
    'drop-agg': 'DropAgg', 'dropagg': 'DropAgg',
    'drop-sens': 'DropSens', 'dropsens': 'DropSens',
}

format_activation_name = {
    'identiy': 'Identity',
    'relu': 'ReLU',
    'elu': 'ELU',
    'sigmoid': 'Sigmoid',
    'tanh': 'Tanh',
}

class FormatEpoch:

    def __init__(self, n_epochs: int):
        self.adj_len = len(str(n_epochs))

    def __call__(self, epoch: int):
        return str(epoch).rjust(self.adj_len, '0')