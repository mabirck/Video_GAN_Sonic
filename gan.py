import torch
from torch import nn
from model import discriminator, generator
# import matplotlib.pyplot as plt


class GAN(nn.Module):

    def __init__(self, num_inputs):
        super(GAN, self).__init__()
        self.G = generator()
        self.D = discriminator(num_inputs)

    def forward(self, inputs):
        image = self.G(inputs)
        return self.D(image)

    def cuda(self):
        self.D.cuda()
        self.G.cuda()
