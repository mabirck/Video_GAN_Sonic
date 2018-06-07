from torch import nn
from model import discriminator
import matplotlib.pyplot as plt

class GAN(nn.Module):

    def __init__(self, num_inputs):
        super(GAN, self).__init__()
        self.D = discriminator(num_inputs)
        self.G = None

    def forward(self, inputs):
        image = self.G(inputs)
        plt.imshow(image)
        plt.show()
        return self.D(image)
