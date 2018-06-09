import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # parser.add_argument('--num_inputs', type=int, default=1,
    #                     help='num_inputs into the model')
    parser.add_argument('--replay_size', default=int(1e6), type=int,
                        help='Buffer Size to train GAN')
    parser.add_argument('--gan_num_epochs', type=int, default=100,
                        help='Epochs to train GAN')
    parser.add_argument('--gan_num_steps', type=int, default=int(1e4),
                            help='Steps por epoch in GAN')
    parser.add_argument('--noise_inputs', type=int, default=62,
                        help='Noise inputs into generative model')
    parser.add_argument('--burn_in_frames', type=int, default=int(100),
                        help='Epochs to train GAN')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size to the model')
    parser.add_argument('--gan_learning_rate', type=float, default=float(2e-4),
                        help='Learning Rate to train Gan')
    parser.add_argument('--num_stack', type=int, default=4,
                        help='Number of frames being stacked')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of simultaneous emulators')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
