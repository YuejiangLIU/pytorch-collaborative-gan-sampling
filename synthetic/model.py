import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, nhidden):
        super(Generator, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(2, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, 2),
        )

    def forward(self, ipt):
        opt = self.block(ipt)
        return opt

class Discriminator(nn.Module):

    def __init__(self, nhidden):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(2, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(True),
            nn.Linear(nhidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, ipt):
        opt = self.block(ipt)
        return opt.squeeze()
