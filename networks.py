import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, latent_dim, model_dim, channels):
        super(Generator, self).__init__()

        def linear(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat), nn.LeakyReLU(0.2, inplace=True)]
            return layers

        def block(in_feat, out_feat, kernel_size=4, stride=2, padding=1):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return layers

        self.channels = channels
        self.latent_dim = latent_dim
        self.model_dim = model_dim

        self.linear = nn.Sequential(*linear(self.latent_dim, 4 * 4 * 4 * self.model_dim))
        self.model_body = nn.Sequential(
            *block(4 * self.model_dim, 2 * self.model_dim),
            *block(2 * self.model_dim, 2 * self.model_dim),
            *block(2 * self.model_dim, self.model_dim),
            nn.Conv2d(self.model_dim, channels, 3, 1, 1),
            nn.Tanh())

    def forward(self, z):
        output = self.linear(z)
        output = torch.reshape(output, [-1, 4*self.model_dim, 4, 4])
        output = self.model_body(output)
        return output


class Critic(nn.Module):
    def __init__(self, model_dim, output_dim, channels):
        super(Critic, self).__init__()

        self.model_dim = model_dim

        def block(in_feat, out_feat, kernel_size=4, stride=2, padding=1, normalize=False):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_body = nn.Sequential(*block(channels, self.model_dim, kernel_size=3, padding=1,
                                               stride=1, normalize=False),
                                        *block(self.model_dim, 2*self.model_dim),
                                        *block(2*self.model_dim, 2*self.model_dim),
                                        *block(2*self.model_dim, 4*self.model_dim))
        self.linear = nn.Linear(4*4*4*self.model_dim, output_dim)
        self.model_tail = nn.Sequential(nn.Dropout(0.4), nn.Linear(output_dim, 1))
        self.sigmoid = nn.Sigmoid()
        self.channels = channels

    def forward(self, img):
        output = self.model_body(img)
        output = torch.reshape(output, [-1, 4*4*4*self.model_dim])
        output_vec = self.linear(output)
        output = self.model_tail(output_vec)
        output = self.sigmoid(output)
        return torch.reshape(output, [-1]), output_vec


class InverseGenerator(nn.Module):
    def __init__(self, latent_dim, model_dim, channels, r=1):
        super(InverseGenerator, self).__init__()

        self.model_dim = model_dim
        self.r = r

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_body = nn.Sequential(*block(channels, self.model_dim, normalize=False),
                                        *block(self.model_dim, 2*self.model_dim),
                                        *block(2*self.model_dim, 4*self.model_dim))
        self.linear = nn.Linear(4*4*4*self.model_dim, latent_dim)
        self.channels = channels

    def forward(self, img):
        output = self.model_body(img)
        output = torch.reshape(output, [-1, self.channels*4*4*4*self.model_dim])
        output = self.linear(output)
        d = torch.norm(output, dim=1, keepdim=True)
        normalized = (output * self.r) / d
        return output, normalized
