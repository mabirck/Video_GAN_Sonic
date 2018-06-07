import torch
from gan import GAN
from arguments import get_args
from sonic_util import make_env
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize


def Tensor(x):
    return torch.FloatTensor(x)


def train_GAN(tranfer_GAN, envs, args):
    obs = envs.reset()

    while(True):
        envs.render()
        actions = envs.action_space.sample()
        print(actions)
        obs, rewards, dones, info = envs.step([actions])
        print(obs.shape)
        prob = tranfer_GAN(Tensor(obs))
        print(prob)


def build_envs(args):
    envs = [make_env]
    args.num_processes = len(envs)

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    args.num_inputs = envs.observation_space.shape[0]

    return envs, args


def main():
    args = get_args()

    envs, args = build_envs(args)

    transfer_GAN = GAN(args.num_inputs)
    print(transfer_GAN)

    train_GAN(transfer_GAN, envs, args)


if __name__ == "__main__":
    main()
