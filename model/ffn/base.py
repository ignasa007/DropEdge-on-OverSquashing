from typing import Union
from torch import Tensor, BoolTensor
from torch.nn import Module, Linear, Sequential
from metrics import Classification, Regression


class BaseHead(Module):

    def __init__(self, task_name: str, layer_sizes: list, activation: Module):

        super(BaseHead, self).__init__()

        formatted_name = task_name.lower()

        if formatted_name == 'classification':
            self.metrics = Classification()
        elif formatted_name == 'regression':
            self.metrics = Regression()
        else:
            raise ValueError('Parameter `task_name` not identified.' \
                f'Expected `classification` or `regression`, but got `{task_name}`.')
        
        module_list = []
        for in_channels, out_channels in zip(layer_sizes[:-1], layer_sizes[1:]):
            module_list.append(Linear(
                in_features=in_channels,
                out_features=out_channels,
                bias=True
            ))
            module_list.append(activation)

        # the output layer does not use any activation
        self.ffn = Sequential(*module_list[:-1])

    def forward(self, node_repr: Tensor, target: Tensor, mask: Union[BoolTensor, None] = None):

        '''
        Process the node embeddings and compute the loss plus any other metrics.
        
        Args:
            node_repr: node representations as returned by the model.
            target: true labels.
            mask: specify indices to compute the metrics over (relevant for node-level tasks).
        '''

        raise NotImplementedError