from torch.nn.functional import dropout
from .base import BaseDropout


class DropNode(BaseDropout):

    def __init__(self, dropout_prob=0.):

        super(DropNode, self).__init__(dropout_prob)

    def apply_message_mat(self, messages, training=True):
        
        return dropout(messages, self.dropout_prob, training=training)