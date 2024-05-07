from torch.nn.functional import dropout
from .base import BaseDropout


class DropMessage(BaseDropout):

    def __init__(self, dropout_prob=0.):

        super(DropMessage, self).__init__(dropout_prob)

    def apply_message_mat(self, messages, training=True):
        
        return dropout(messages, self.dropout_prob, training=training)