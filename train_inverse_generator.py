import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from networks import Generator, InverseGenerator

parser = argparse.ArgumentParser(description='Inverse generator training on FashionMNIST dataset.')
parser.add_argument('--data_path', default='data/mnist/', type=str, help='Directory of the dataset')
parser.add_argument('--pretrained_models_path', default='pretrained_models',
                    type=str, help='Directory for trained models')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--latent_dim', default=100, type=int, help='dimensions of latent vector')
parser.add_argument('--gan_type', default='dcgan', type=str, choices=['wgan', 'dcgan'], help="dcgan or wgan")
parser.add_argument('--train_set_size', default=60000, type=int, help="size of training dataset")
parser.add_argument('--test_set_size', default=10000, type=int, help="size of test dataset")
args = parser.parse_args()

PRETRAINED_GEN_PATH = os.path.join(args.pretrained_models_path, args.gan_type, f"{args.gan_type}_g.pkl")

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def evaluate(generator, inv_generator, criterion, eval_dl):
    losses = []
    inv_generator.eval()
    for i, z in enumerate(eval_dl):
        images = generator(z)

        z_inv = inv_generator(images)
        loss = criterion(z, z_inv)
        loss.backward()

        losses.append(loss.data.item())
    return np.mean(losses)


def train(generator, inv_generator, optimizer, criterion, train_dl, val_dl):
    train_losses = []
    valid_losses = []
    generator.eval()
    for epoch in range(1, args.epochs+1):
        losses = []
        inv_generator.train()
        for i, z in enumerate(train_dl):
            images = generator(z)

            optimizer.zero_grad()
            z_inv = inv_generator(images)
            loss = criterion(z, z_inv)
            loss.backward()
            optimizer.step()

            losses.append(loss.data.item())

        train_losses.append(np.mean(losses))
        valid_losses.append(evaluate(generator, inv_generator, criterion, val_dl))
        print(
            f"[epoch {epoch:d}/{args.epochs:d}] [train loss: {train_losses[-1]:f}] [valid loss: {valid_losses[-1]:f}]"
        )
    return train_losses, valid_losses
