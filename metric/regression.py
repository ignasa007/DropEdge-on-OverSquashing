from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
from .base import Metrics


class Regression(Metrics):

    def __init__(self):

        super(Regression, self).__init__()

        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_absolute_percentage_error = MeanAbsolutePercentageError()
        self.mean_squared_error = MeanSquaredError()
    
    def compute(self, input: Tensor, target: Tensor):

        input = input.squeeze()
        
        if input.ndim > 1:
            raise ValueError(f'Expected predictions to be up to 2-dimensional (got shape {input.shape}).')
        
        mae_loss = self.mean_absolute_error(input, target)
        mape_loss = self.mean_absolute_percentage_error(input, target)
        mse_loss = self.mean_squared_error(input, target)

        result = {
            'Mean Absolute Error': mae_loss,
            'Mean Absolute Percentage Error': mape_loss,
            'Mean Squared Error': mse_loss,
        }

        return result