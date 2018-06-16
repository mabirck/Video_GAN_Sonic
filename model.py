import torch
import utils
import torch.nn as nn
from utils import init, init_normc_
from distributions import Categorical, DiagGaussian


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy):
        super(Policy, self).__init__()
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], recurrent_policy)
        elif len(obs_shape) == 1:
            assert not recurrent_policy, \
                "Recurrent policy is not implemented for the MLP controller"
            self.base = MLPBase(obs_shape[0])
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.state_size = self.base.state_size

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


class discriminator(nn.Module):
    def __init__(self, num_inputs):
        super(discriminator, self).__init__()
        self.output_dim = 1
        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, self.output_dim)),
            nn.Sigmoid()
        )

        self.train()

    def forward(self, inputs):
        x = self.main(inputs)
        return x

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(generator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 84
            self.input_width = 84
            self.input_dim = 62
            self.output_dim = 4
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 62
            self.output_dim = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)
        return x


class AdvGenerator(nn.Module):
    def __init__(self):
        super(AdvGenerator, self).__init__()
        self.input_height = 84
        self.input_width = 84
        #self.input_dim = 62
        self.output_dim = 4

        self.feature_map_sizes = [self.output_dim, 128, 256, 512, 256, 128, 1]
        self.kernel_sizes = [7, 5, 5, 5, 5, 7]

        self.deconv = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.deconv(input)
        #print(x.size(), "THIS IS MY GENERATOR")
        return x

class AdvDiscriminator(nn.Module):
    def __init__(self, args):
        super(AdvDiscriminator, self).__init__()
        self.args = args
        self.input_height = 84
        self.input_width = 84
        #self.input_dim = 62
        self.output_dim = 4

        self.feature_map_sizes = [self.output_dim, 128, 256, 512, 128]
        self.kernel_sizes = [7, 7, 5, 5]
        self.fc_sizes = [1024, 512]

        self.conv = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 256, kernel_size=7),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 512, kernel_size=5),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )

        self.sequential = nn.Sequential(
            nn.Linear(12 * 12 * 128, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        utils.initialize_weights(self)


    def forward(self, input):
        #print(input.size(), "DISCRIMINATOR INPUT")
        x = self.conv(input)
        #print(x.size(), "DISCRIMINATOR CONV")
        x = x.view(self.args.batch_size, -1)
        return self.sequential(x)


class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return self.critic_linear(x), x, states


class MLPBase(nn.Module):
    def __init__(self, num_inputs):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(64, 1))

        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor, states


class RandomPolicy(object):
    def __init__(self, env):
        self.env = env

    def get_action(self):
        return [self.env.action_space.sample()]
