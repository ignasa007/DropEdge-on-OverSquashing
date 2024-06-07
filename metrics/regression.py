from torch import Tensor
from torchmetrics import MeanSquaredError


class Regression:

    def __init__(self, num_classes: int):

        # TODO: problem is that we want to compute the metrics dimension-wise
        #   in case of multi-ouput regression (like QM9)
        # TODO: also gotta think about logging in this case

        super(Regression, self).__init__()
        self.mean_squared_error = MeanSquaredError()

    def reset(self):

        self.mean_squared_error.reset()
    
    def update_mse(self, input: Tensor, target: Tensor):

        input = input.reshape(target.shape)
        mse = self.mean_squared_error.forward(input, target)
        return mse
    
    def compute_mse(self):

        mean_sq_error = self.mean_squared_error.compute()
        self.reset()

        return mean_sq_error