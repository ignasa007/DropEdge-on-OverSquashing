from torch import Tensor


class Metrics:

    def __init__(self):

        pass

    def reset(self):

        raise NotImplementedError
    
    def update(self, input: Tensor, target: Tensor):

        raise NotImplementedError
    
    def compute(self):

        raise NotImplementedError