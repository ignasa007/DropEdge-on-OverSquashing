root = '../data'
batch_size = 64

from enum import Enum

class Splits(Enum):

    train_split = 0.60
    val_split = 0.16
    test_split = 0.20