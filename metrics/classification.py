from torch import sigmoid, Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC, \
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC
from metrics.base import Metrics


class Classification(Metrics):

    def __init__(self, num_classes: int):

        super(Classification, self).__init__()

        if not isinstance(num_classes, int):
            raise TypeError(f'Expected `num_classes` to be an instance of `int` (got {type(num_classes)}).')
        
        # TODO: update + compute for cross-entropy loss. torch.nn classes don't support.

        if num_classes == 2:
            self.loss_fn = BCEWithLogitsLoss(reduction=None)
            self.accuracy_fn = BinaryAccuracy()
            self.f1score_fn = BinaryF1Score()
            self.auroc_fn = BinaryAUROC()

        elif num_classes > 2:
            self.loss_fn = CrossEntropyLoss(reduction=None)
            self.accuracy_fn = MulticlassAccuracy()
            self.f1score_fn = MulticlassF1Score(num_classes)
            self.auroc_fn = MulticlassAUROC(num_classes)

        else:
            raise ValueError(f'Expected `num_classes` to be >1 (got {num_classes}).')
        
    def reset(self):

        self.loss_fn.reset()
        self.accuracy_fn.reset()
        self.f1score_fn.reset()
        self.auroc_fn.reset()
    
    def update(self, input: Tensor, target: Tensor):

        input = input.reshape(target.shape)
        preds = sigmoid(input)

        self.loss_fn.update(preds, target)
        self.accuracy_fn.update(preds, target)
        self.f1score_fn.update(preds, target)
        self.auroc_fn.update(preds, target)

    def compute(self):

        cross_entropy = self.loss_fn.compute()
        accuracy = self.accuracy_fn.compute().item()
        f1_score = self.f1score_fn.compute().item()
        auroc = self.auroc_fn.compute().item()

        self.reset()

        metrics = [
            ('Cross Entropy Loss', cross_entropy.item()),
            ('Accuracy', accuracy),
            ('F1 Score', f1_score),
            ('AU-ROC', auroc),
        ]

        return cross_entropy, metrics