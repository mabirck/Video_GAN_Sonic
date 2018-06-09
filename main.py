import torch
from gan import GAN
from arguments import get_args
from sonic_util import make_env
from torch.autograd import Variable as V
from utils import plot_loss, plot_result
from replay_memory import ReplayMemory, samples_to_tensors as ToTensor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

betas = (0.5, 0.999)

def Tensor(x):
    return torch.FloatTensor(x)


def Noise(args):
    # fixed noise
    if args.cuda:
        noise = V(torch.rand((args.batch_size, args.noise_inputs)).cuda(), volatile=True)
    else:
        noise = V(torch.rand((args.batch_size, args.noise_inputs)), volatile=True)
    return noise


def train_GAN(tranfer_GAN, envs, replay_buffer, args):
    fixed_noise = Noise(args)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Optimizers
    G_optimizer = torch.optim.Adam(tranfer_GAN.G.parameters(), lr=args.gan_learning_rate, betas=betas)
    D_optimizer = torch.optim.Adam(tranfer_GAN.D.parameters(), lr=args.gan_learning_rate, betas=betas)

    # Training GAN
    D_avg_losses = []
    G_avg_losses = []

    replay_buffer.burn_in(envs, args.burn_in_frames)

    # Reset Env Before Start
    envs.reset()

    for epoch in range(args.gan_num_epochs):
        D_losses = []
        G_losses = []

        for step in range(args.gan_num_steps):
            # Feed Memory Replay with Real Sonic Images #
            envs.render()
            actions = envs.action_space.sample()
            obs, rewards, dones, info = envs.step([actions])
            print(obs.shape)
            replay_buffer.append(obs.squeeze(0))
            print(len(replay_buffer.samples))
            # __________________________________________________#

            # Train discriminator with real data #
            real_x = V(ToTensor(replay_buffer.sample(args.batch_size))).squeeze() # labels
            real_y = V(torch.ones(real_x.size()[0]).cuda())
            D_probs = tranfer_GAN.D(real_x)
            D_real_loss = criterion(D_probs, real_y)
            # __________________________________________________#

            # Get noise to Feed Generator #
            noise = Noise(args)
            fake_image = tranfer_GAN.G(noise)
            fake_y = V(torch.zeros(fake_image.size()[0]).cuda())

            D_fake_probs = tranfer_GAN.D(fake_image).squeeze()
            D_fake_loss = criterion(D_fake_probs, fake_y)

            # Back propagation
            D_loss = D_real_loss + D_fake_loss
            tranfer_GAN.D.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            # _________________________________________________#

            # Train generator
            noise = Noise(args)
            fake_image = tranfer_GAN.G(noise)

            D_fake_probs = tranfer_GAN.D(fake_image)
            G_loss = criterion(D_fake_probs, real_y)

            # Back propagation
            tranfer_GAN.D.zero_grad()
            tranfer_GAN.G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            D_losses.append(D_loss.data[0])
            G_losses.append(G_loss.data[0])
            # _________________________________________________#

            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                   % (epoch+1, args.gan_num_epochs, step+1, len(replay_buffer.samples), D_loss.data[0], G_loss.data[0]))

        D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
        G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

        # avg loss values for plot
        D_avg_losses.append(D_avg_loss)
        G_avg_losses.append(G_avg_loss)

        plot_loss(D_avg_losses, G_avg_losses, epoch, save=True)

        # Show result for fixed noise
        plot_result(tranfer_GAN.G, fixed_noise, epoch, save=True, fig_size=(5, 5))

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

    transfer_GAN = GAN(args.num_stack)

    if args.cuda:
        transfer_GAN.cuda()
    print(transfer_GAN)

    train_GAN(transfer_GAN, envs, replay_buffer, args)


if __name__ == "__main__":
    main()
