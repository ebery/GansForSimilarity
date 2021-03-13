import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from networks import Generator, Critic
import numpy as np


def __weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_models(mode, latent_dim, model_dim, device, init=True):
    generator = Generator(latent_dim, model_dim).to(device)
    critic = Critic(mode, model_dim).to(device)
    if init:
        generator.apply(__weights_init_normal)
        critic.apply(__weights_init_normal)
    return generator, critic


def get_dataloader(data_path, img_size, batch_size):
    os.makedirs(data_path, exist_ok=True)
    transforms_list = [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            data_path,
            train=True,
            download=True,
            transform=transforms.Compose(transforms_list),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader


def plot_losses(mode):
    g_path = f"Q4 Results/{mode}_losses_g.csv"
    d_path = f"Q4 Results/{mode}_losses_d.csv"
    if not os.path.exists:
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
    plt.savefig(f"Q4 Results/{mode}_loss.png")
