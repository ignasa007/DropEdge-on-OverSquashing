from typing import Union
from torch import Tensor, BoolTensor
from torch.nn import Module, Linear, Sequential
from metrics import Classification, Regression


class BaseHead(Module):

    def __init__(self, task_name: str, num_classes: int, layer_sizes: list, activation: Module):

        super(BaseHead, self).__init__()

        formatted_name = task_name.lower()

        output_dim = num_classes
        if formatted_name == 'classification':
            self.metrics = Classification(num_classes)
            if num_classes == 2: output_dim = 1
        elif formatted_name == 'regression':
            self.metrics = Regression(num_classes)
        else:
            raise ValueError('Parameter `task_name` not identified.' \
                f'Expected `classification` or `regression`, but got `{task_name}`.')
        
        layer_sizes = layer_sizes + [output_dim]
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

    def preprocess(self, node_repr: Tensor, target: Tensor, mask: Union[Tensor, BoolTensor, None] = None):

        '''
        Preprocess the input:
            - for node-level tasks, filter out the embeddings and corresponding labels using $mask.
            - for graph-level tasks, compute the mean of the node embeddings from each graph.
        '''

        raise NotImplementedError

    def forward(self, node_repr: Tensor, target: Tensor, mask: Union[Tensor, BoolTensor, None] = None):

        '''
        Process the node embeddings and compute the loss plus any other metrics.
        
        Args:
            node_repr: node representations as returned by the model.
            target: true labels.
            mask: 
                - for node-level tasks, specify indices to compute the metrics over.
                - for graph-level tasks, specify node sizes for the batch of graphs. 
        '''

        node_repr, target = self.preprocess(node_repr, target, mask)
        
        out = self.ffn(node_repr)
        loss = self.metrics.update(out, target)

        return loss