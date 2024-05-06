from torch.nn.functional import dropout
from .base import BaseDropout


class Dropout(BaseDropout):

    def __init__(self, dropout_prob=0.):

        super(Dropout, self).__init__(dropout_prob)

    def apply_feature_mat(self, x, training=True):
        
        return dropout(x, self.dropout_prob, training=training)