import torch.nn as nn


class Temp(nn.Module):

    def __init__(self):

        super().__init__()
        self.drop = nn.Dropout1d()

    def forward(self):

        pass


t = Temp()
t.train()
print(t.training, t.drop.training)
t.drop.training = True
t.train(False)
print(t.training, t.drop.training)
t.drop.training = True
print(t.training, t.drop.training)