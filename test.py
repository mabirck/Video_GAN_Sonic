import torch
from gan import GAN
from arguments import get_args
from sonic_util import make_env
from torch.autograd import Variable as V
from replay_memory import samples_to_tensors as ToTensor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import matplotlib.pyplot as plt

def test_GAN(transfer_GAN, envs, args):

    # Reset Env Before Start
    obs = envs.reset()
    for img in obs[0]:
        print(img.shape)
        plt.imshow(img)
        plt.show()

    print(torch.load("model.pt"))
    transfer_GAN = torch.load("model.pt")

    for steps in range(60):
        print(steps)
        obs = V(torch.FloatTensor(obs))
        fake_image = transfer_GAN.G(obs)
        print(fake_image.squeeze().data.cpu().numpy().shape)
        plt.imshow(fake_image.squeeze().data.cpu().numpy())
        plt.show()
        obs[:,-1:,:,:] = fake_image


def build_envs(args):
    envs = [make_env]
    args.num_processes = len(envs)

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    return envs, args

def main():
    args = get_args()

    envs, args = build_envs(args)

    transfer_GAN = GAN(args.num_stack)

    if args.cuda:
        transfer_GAN.cuda()
    print(transfer_GAN)

    test_GAN(transfer_GAN, envs, args)


if __name__ == "__main__":
    main()
