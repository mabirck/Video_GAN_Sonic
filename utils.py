import os
import gym
import csv
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as V
import matplotlib.pyplot as plt
from gym.spaces.box import Box



# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

def initialize_weights(net):
    for m in net.modules():
        print(m)
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def observation(self, observation):
        # TODO: Check this workaround
        observation = np.squeeze(np.array(observation._frames))
        return observation

# Plot losses
def save_loss(d_losses, g_losses, num_epoch, save=False, save_dir='Sonic_VGAN_results/', show=False):

    path = save_dir+'log.txt'
    #path += "_".join([args.arc, str(args.epochs), args.filter_reg, str(args.phi), 'seed', str(args.seed), 'depth', str(args.depth), args.intra_extra])
    #path = path+'.done.csv' if epoch == args.epochs else path+'.csv'

    assert not(os.path.isfile(path) == True and num_epoch == 0), "That can't be right. This file should not be here!!!!"
    fields = [num_epoch, "d_losses", d_losses[-1], "g_losses", g_losses[-1]]


    with open(path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


# Plot losses
def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='Sonic_VGAN_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'CelebA_DCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, noise, num_epoch, save=False, save_dir='CelebA_DCGAN_results/', show=False, fig_size=(5, 5)):
    generator.eval()

    noise = V(noise.cuda(), volatile=True)
    gen_image = generator(noise)
    gen_image = denorm(gen_image)

    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        # Scale to 0-255
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        # ax.imshow(img.cpu().data.view(image_size, image_size, 3).numpy(), cmap=None, aspect='equal')
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'CelebA_DCGAN_epoch_{:d}'.format(num_epoch+1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def lp_loss(fake_frames, real_frames, l_num=2):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.
    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @return: The lp loss.
    """
    return torch.sum(torch.abs(V(fake_frames) - V(real_frames))**l_num)
