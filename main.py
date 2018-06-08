import torch
from torch.autograd import Variable as V
from gan import GAN
from arguments import get_args
from sonic_util import make_env
from replay_memory import ReplayMemory
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def Tensor(x):
    return torch.FloatTensor(x)


def Noise(args):
    # fixed noise
    if args.cuda:
        noise = V(torch.rand((1 * args.num_stack, args.noise_inputs)).cuda(), volatile=True)
    else:
        noise = V(torch.rand((1 * args.num_stack, args.noise_inputs)), volatile=True)
    return noise


def train_GAN(tranfer_GAN, envs, replay_buffer, args):
    obs = envs.reset()

    while(True):

        # Feed Memory Replay with Real Sonic Images
        envs.render()
        actions = envs.action_space.sample()
        obs, rewards, dones, info = envs.step([actions])
        ################################################

        # Get noise to Feed Generator #
        noise = Noise(args)
        prob = tranfer_GAN(noise)

        print(prob)


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

    replay_buffer = ReplayMemory(args.replay_size)
    envs, args = build_envs(args)

    transfer_GAN = GAN(args.num_inputs)

    if args.cuda:
        transfer_GAN.cuda()
    print(transfer_GAN)

    train_GAN(transfer_GAN, envs, replay_buffer, args)


if __name__ == "__main__":
    main()
