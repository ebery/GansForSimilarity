import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, latent_dim, model_dim):
        super(Generator, self).__init__()

        def linear(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat), nn.BatchNorm1d(out_feat), nn.ReLU(inplace=True)]
            return layers

        def block(in_feat, out_feat):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            return layers

        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.linear = nn.Sequential(*linear(self.latent_dim, 4*4*4*self.model_dim))
        self.model_body = nn.Sequential(*block(4*self.model_dim, 2*self.model_dim),
                                        *block(2*self.model_dim, self.model_dim),
                                        nn.ConvTranspose2d(self.model_dim, 1, 4, 2, 1),
                                        nn.Tanh())

    def forward(self, z):
        output = self.linear(z)
        output = torch.reshape(output, [-1, 4*self.model_dim, 4, 4])
        output = self.model_body(output)
        return output


class Critic(nn.Module):
    def __init__(self, mode, model_dim, output_dim):
        super(Critic, self).__init__()

        self.model_dim = model_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_body = nn.Sequential(*block(1, self.model_dim, normalize=False),
                                        *block(self.model_dim, 2*self.model_dim),
                                        *block(2*self.model_dim, 4*self.model_dim))
        self.linear = nn.Linear(4*4*4*self.model_dim, output_dim)
        self.model_tail = nn.Sequential(nn.Dropout(0.5), nn.Linear(output_dim, 1))
        self.sigmoid = nn.Sigmoid()
        self.mode = mode

    def forward(self, img):
        output = self.model_body(img)
        output = torch.reshape(output, [-1, 4*4*4*self.model_dim])
        output_vec = self.linear(output)
        output = self.model_tail(output_vec)
        if self.mode == 'dcgan':
            output = self.sigmoid(output)
        return torch.reshape(output, [-1]), output_vec
