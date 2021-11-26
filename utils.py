import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, random_split
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


def get_models(latent_dim, model_dim, device, output_dim, channels, init=True):
    generator = Generator(latent_dim, model_dim, channels).to(device)
    critic = Critic(model_dim, output_dim, channels).to(device)
    if init:
        generator.apply(__weights_init_normal)
        critic.apply(__weights_init_normal)
    return generator, critic


def sample_spherical_distribution(num_samples, latent_dim, device, r):
    samples = Variable(torch.randn(num_samples, latent_dim, device=device))
    d = torch.norm(samples, dim=1, keepdim=True)
    samples = (samples * r) / d
    return samples


def get_dataloader(data_path, img_size, batch_size, train=True, validation=False, length=None):
    os.makedirs(data_path, exist_ok=True)
    transforms_list = [transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(),
                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    dataset = datasets.CIFAR10(
            data_path,
            train=train,
            download=True,
            transform=transforms.Compose(transforms_list),
        )
    if validation:
        dataset, val_set = random_split(dataset, length)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    if validation:
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=True,
        )
        return dataloader, val_loader
    return dataloader, None


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


def angular_loss(x, y, r=1):
    cos_theta = torch.sum(x * y, dim=1)
    cos_theta = torch.clamp(cos_theta, -1, 1)
    theta = torch.acos(cos_theta)
    pi = torch.acos(torch.zeros(1)).data.item()*2
    partial = theta / pi
    perimeter = 2 * pi * r
    loss = partial * perimeter
    return torch.mean(loss)


def sphere_loss(x, r):
    return torch.mean(((torch.norm(x, p=2, dim=1) - r) ** 2))


def plot_losses(results_dir):
    g_path = os.path.join(results_dir, f"losses_g.csv")
    d_path = os.path.join(results_dir, f"losses_d.csv")
    if not os.path.exists(g_path):
        print(f"model is not trained yet. Please run training models section first")
        return
    g_losses, d_losses = np.genfromtxt(g_path), np.genfromtxt(d_path)
    plt.figure()
    plt.plot(list(range(len(g_losses))), g_losses, list(range(len(d_losses))), d_losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(f"loss vs iterations")
    plt.legend(["G", "D"])
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"loss.png"))


def create_dirs(dirs_path_list):
    for dir in dirs_path_list:
        os.makedirs(dir, exist_ok=True)


def get_optimizers(generator, critic, dcgan_lr, betas=None):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=dcgan_lr, betas=betas)
    optimizer_D = torch.optim.Adam(critic.parameters(), lr=dcgan_lr, betas=betas)
    return optimizer_G, optimizer_D
