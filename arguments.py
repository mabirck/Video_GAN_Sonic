import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--num_inputs', type=int, default=1,
                        help='num_inputs into the model')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of simultaneous emulators')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
