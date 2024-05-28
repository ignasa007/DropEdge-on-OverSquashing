from typing import Union
from torch import Tensor, BoolTensor
from torch.nn import Module
from ..metric import Classification, Regression


class BaseHead(Module):

    def __init__(self, task_name: str):

        super(BaseHead, self).__init__()

        formatted_name = task_name.lower()

        if formatted_name == 'classification':
            self.metrics = Classification()
        elif formatted_name == 'regression':
            self.metrics = Regression()
        else:
            raise ValueError(f'Task not identified. Expected `classification` or `regression`, but got `{task_name}`.')

    # TODO: How are graph level tasks performed? What is done after message passing?
    def forward(self, node_repr: Tensor, labels: Tensor, mask: Union[BoolTensor, None] = None):

        '''
        Process the node embeddings and compute the loss plus any other metrics.
        
        Args:
            node_repr: node representations as returned by the model.
            labels: true labels.
            mask: specify indices to compute the metrics over (relevant for node level tasks).
        '''

        raise NotImplementedError