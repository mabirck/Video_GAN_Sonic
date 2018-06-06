import gym
import torch
from gan import GAN
from arguments import get_args



def train_GAN(tranfer_GAN):
    env = gym.make("BreakoutDeterministic-v4")

    obs = torch.tensor(env.reset().resize(84, 84, 3))
    print(obs.size())

    prob = tranfer_GAN(obs)
    print(prob)


def main():
    args = get_args()

    transfer_GAN = GAN(args.num_inputs)
    print(transfer_GAN)

    train_GAN(transfer_GAN)


if __name__ == "__main__":
    main()
