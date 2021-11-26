from datasets.FMNistSimilarityDataset import FMNistSimilarityDataset
from torch.utils.data import DataLoader
import utils as utl
import argparse
import torch
from torchvision.transforms import transforms
from torch import nn
from torchvision.utils import save_image
from networks import Generator, InverseGenerator, Critic
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Testing similarity')
parser.add_argument('--data_path', default='data/mnist/', type=str, help='Directory of the dataset')
parser.add_argument('--pretrained_models_path', default='pretrained_models',
                    type=str, help='Directory for trained models')
parser.add_argument('--results_path', default='results/', type=str, help='Results directory')
parser.add_argument('--latent_dim', default=100, type=int, help='dimensions of latent vector')
parser.add_argument('--gan_type', default='dcgan', type=str, choices=['wgan', 'dcgan'], help="dcgan or wgan")
args = parser.parse_args()

IMG_SIZE = 32
DIM = 64
SAMPLES_PER_CLASS = 100
NUM_CLASSES = 10

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


transforms_list = [transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
dataset = FMNistSimilarityDataset(args.data_path, transforms.Compose(transforms_list), SAMPLES_PER_CLASS,
                                  NUM_CLASSES).get_dataset()
inv_generator = InverseGenerator(args.latent_dim, DIM)
inv_generator.load_state_dict(torch.load())