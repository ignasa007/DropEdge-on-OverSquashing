from torch import Tensor


class Metrics:

    def __init__(self):

        pass

    def __call__(self, *args, **kwargs):

        return self.compute(*args, **kwargs)
    
    def compute(self, input: Tensor, target: Tensor):

        raise NotImplementedError