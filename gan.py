from torch import nn
from model import CNN_D


class GAN(nn.Module):

    def __init__(self, num_inputs):
        super(GAN, self).__init__()
        self.D = CNN_D(num_inputs)
        self.G = None

    def forward(self, inputs):
        return self.D(inputs)
