import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from networks import Generator, Critic
import numpy as np


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def __weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_models(mode, latent_dim, model_dim, device, output_dim, init=True):
    generator = Generator(latent_dim, model_dim).to(device)
    critic = Critic(mode, model_dim, output_dim).to(device)
    if init:
        generator.apply(__weights_init_normal)
        critic.apply(__weights_init_normal)
    return generator, critic


def get_dataloader(data_path, img_size, batch_size, train=True):
    os.makedirs(data_path, exist_ok=True)
    transforms_list = [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            data_path,
            train=train,
            download=True,
            transform=transforms.Compose(transforms_list),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader


def wasserstein_loss(fake_imgs, critic, device, real_imgs=None):
    gen_loss = -torch.mean(critic(fake_imgs)[0]).to(device)
    if real_imgs is None:
        return gen_loss
    return -(torch.mean(critic(real_imgs)[0])+gen_loss)


def adversarial_loss(fake_imgs, critic, device, real_imgs=None):
    dcgan_loss = torch.nn.BCELoss()
    valid = Variable(Tensor(fake_imgs.shape[0]).fill_(1.0), requires_grad=False)
    if real_imgs is None:
        gen_cost = dcgan_loss(critic(fake_imgs)[0], valid)
        return gen_cost.to(device)
    fake = Variable(Tensor(fake_imgs.shape[0]).fill_(0.0), requires_grad=False)
    critic_cost = dcgan_loss(critic(fake_imgs.detach())[0], fake) + dcgan_loss(critic(real_imgs)[0], valid)
    return critic_cost.to(device) / 2


def plot_losses(results_dir, mode):
    g_path = os.path.join(results_dir, f"{mode}_losses_g.csv")
    d_path = os.path.join(results_dir, f"{mode}_losses_d.csv")
    if not os.path.exists(g_path):
        print(f"{mode} model is not trained yet. Please run training models section first")
        return
    g_losses, d_losses = np.genfromtxt(g_path), np.genfromtxt(d_path)
    plt.figure()
    plt.plot(list(range(len(g_losses))), g_losses, list(range(len(d_losses))), d_losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(f"{mode} loss vs iterations")
    plt.legend(["G", "D"])
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{mode}_loss.png"))


def create_dirs(dirs_path_list):
    for dir in dirs_path_list:
        os.makedirs(dir, exist_ok=True)


def get_optimizers(generator, critic, mode, dcgan_lr, wgan_lr, betas=None):
    if mode == 'wgan':
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=wgan_lr)
        optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=wgan_lr)
    else:
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=dcgan_lr, betas=betas)
        optimizer_D = torch.optim.Adam(critic.parameters(), lr=dcgan_lr, betas=betas)
    return optimizer_G, optimizer_D
