import argparse
import os
import numpy as np
import torch
from datasets.LatentVectorsDataset import LatentVectorsDataset
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pandas as pd

from networks import Generator, InverseGenerator

parser = argparse.ArgumentParser(description='Inverse generator training on FashionMNIST dataset.')
parser.add_argument('--data_path', default='data/latent_vectors/', type=str, help='Directory of the dataset')
parser.add_argument('--pretrained_models_path', default='pretrained_models',
                    type=str, help='Directory for trained models')
parser.add_argument('--results_path', default='results/', type=str, help='Results directory')
parser.add_argument('--generated_images_path', default='Generated Images', type=str, help='generated images directory')
parser.add_argument('--epochs', default=15, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--latent_dim', default=100, type=int, help='dimensions of latent vector')
parser.add_argument('--gan_type', default='dcgan', type=str, choices=['wgan', 'dcgan'], help="dcgan or wgan")
parser.add_argument('--train_set_size', default=60000, type=int, help="size of training dataset")
parser.add_argument('--test_set_size', default=10000, type=int, help="size of test dataset")
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for ADAM')
parser.add_argument('--b1', default=0.5, type=float, help='betta1 for ADAM optimizer')
parser.add_argument('--b2', default=0.999, type=float, help='betta2 for ADAM optimizer')
args = parser.parse_args()

PRETRAINED_GEN_PATH = os.path.join(args.pretrained_models_path, args.gan_type, f"{args.gan_type}_g.pkl")
DIM = 64

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
    delta = 5e-4
    for epoch in range(1, args.epochs+1):
        losses = []
        inv_generator.train()
        for i, z in enumerate(train_dl):
            images = generator(z.to(device))

            optimizer.zero_grad()
            z_inv = inv_generator(images)
            loss = criterion(z, z_inv)
            loss.backward()
            optimizer.step()

            losses.append(loss.data.item())

        val_loss = evaluate(generator, inv_generator, criterion, val_dl)

        # if len(valid_losses) > 0 and valid_losses[-1] - val_loss < delta:
        #     print("changing lr")
        #     for g in optimizer.param_groups:
        #         g['lr'] *= 0.99
        #     delta /= 2

        for g in optimizer.param_groups:
            g['lr'] *= 0.99

        train_losses.append(np.mean(losses))
        valid_losses.append(val_loss)
        print(
            f"[epoch {epoch:d}/{args.epochs:d}] [train loss: {train_losses[-1]:f}] [valid loss: {valid_losses[-1]:f}]"
        )
    model_path = os.path.join(args.pretrained_models_path, args.gan_type, f"{args.gan_type}_inv_g.pkl")
    torch.save(inv_generator.state_dict(), model_path)
    losses = np.column_stack((train_losses, valid_losses))
    df = pd.DataFrame(losses)
    header = ["train", "valid"]
    df.to_csv(os.path.join(args.results_path, f"{args.gan_type}_losses_inv_g.csv"), header=header, index=False)
    return train_losses, valid_losses


def visual_test(generator, inv_generator, test_dl):
    inv_generator.eval()
    z = next(iter(test_dl))
    images = generator(z)[: int(np.floor(np.sqrt(args.batch_size))**2), :]

    z_inv = inv_generator(images)
    inv_images = generator(z_inv)
    results_path = os.path.join(args.results_path, args.generated_images_path, args.gan_type, "Visual Tests")
    os.makedirs(results_path, exist_ok=True)
    save_image(images.data, os.path.join(results_path, "gen_images.png"),
               nrow=int(np.floor(np.sqrt(args.batch_size))), normalize=True)
    save_image(inv_images.data, os.path.join(results_path, "gen_inv_images.png"),
               nrow=int(np.floor(np.sqrt(args.batch_size))), normalize=True)


generator = Generator(args.latent_dim, DIM)
generator.to(device)
generator.load_state_dict(torch.load(PRETRAINED_GEN_PATH))
generator.eval()

inv_generator = InverseGenerator(args.latent_dim, DIM)
inv_generator.to(device)
optimizer = Adam(inv_generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
criterion = nn.MSELoss().to(device)

train_ds = LatentVectorsDataset(os.path.join(args.data_path, "training.pt"))
valid_ds = LatentVectorsDataset(os.path.join(args.data_path, "validation.pt"))
test_ds = LatentVectorsDataset(os.path.join(args.data_path, "test.pt"))
train_dl = DataLoader(train_ds, args.batch_size, num_workers=0)
val_dl = DataLoader(valid_ds, args.batch_size, num_workers=0)
test_dl = DataLoader(test_ds, args.batch_size, num_workers=0)

train_losses, valid_losses = train(generator, inv_generator, optimizer, criterion, train_dl, val_dl)
test_loss = evaluate(generator, inv_generator, criterion, test_dl)
print("Test Loss: ", test_loss)
visual_test(generator, inv_generator, test_dl)
