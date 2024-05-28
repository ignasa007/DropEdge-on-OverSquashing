from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchmetrics import Accuracy
from .base import Metrics


class Classification(Metrics):

    def __init__(self):

        super(Classification, self).__init__()

        self.bce_with_logits_loss = BCEWithLogitsLoss()
        self.cross_entropy_loss = CrossEntropyLoss()
    
    def compute(self, input: Tensor, target: Tensor):

        input = input.squeeze()

        if input.ndim > 2:
            raise ValueError(f'Expected predictions to be up to 2-dimensional (got shape {input.shape}).')
        
        if input.ndim == 1:
            loss_fn = self.bce_with_logits_loss
            accuracy_fn = Accuracy(task='binary')
            preds = (input>0).to(target.dtype)
        else:
            loss_fn = self.cross_entropy_loss
            accuracy_fn = Accuracy(task='multiclass', num_classes=input.size(1))
            preds = input.argmax(dim=1)
        
        loss = loss_fn(input, target)
        accuracy = accuracy_fn(preds, target)

        result = {
            'Cross Entropy Loss': loss,
            'Accuracy': accuracy,
        }

        return result