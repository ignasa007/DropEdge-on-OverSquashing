from typing import Union
from torch import Tensor, BoolTensor
from torch.nn import Module
from ..metrics import Classification, Regression


class BaseHead(Module):

    def __init__(self, task: str):

        super(BaseHead, self).__init__()

        if task == 'classification':
            self.metrics = Classification()
        elif task == 'regression':
            self.metrics = Regression()
        else:
            raise ValueError(f'Task not identified. Expected `classification` or `regression`, but got `{task}`.')

    # TODO: How are graph level tasks performed? What is done after message passing?
    def forward(self, node_repr: Tensor, labels: Tensor, mask: Union[BoolTensor, None] = None):

        '''
        Compute the loss and any other metrics.
        
        Args:
            node_repr: node representations as returned by the model.
            labels: true node labels.
            mask: specify which nodes to compute the metrics over.
        '''

        raise NotImplementedError