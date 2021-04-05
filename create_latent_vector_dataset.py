import torch
import os
import argparse
import numpy as np
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Create latent vector dataset from uniform distribution')
parser.add_argument('--data_path', default='data/latent_vectors/', type=str, help='Directory of the dataset')
parser.add_argument('--train_set_size', default=60000, type=int, help='train set size')
parser.add_argument('--test_set_size', default=10000, type=int, help='test set size')
parser.add_argument('--latent_dim', default=100, type=int, help='dimensions of latent vector')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

np.random.seed(42)
torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)

os.makedirs(args.data_path, exist_ok=True)

train_set = Variable(Tensor(np.random.normal(0, 1, (args.train_set_size, args.latent_dim))))
valid_set = Variable(Tensor(np.random.normal(0, 1, (args.test_set_size, args.latent_dim))))
test_set = Variable(Tensor(np.random.normal(0, 1, (args.test_set_size, args.latent_dim))))

torch.save(train_set, os.path.join(args.data_path, "training.pt"))
torch.save(test_set, os.path.join(args.data_path, "validation.pt"))
torch.save(test_set, os.path.join(args.data_path, "test.pt"))

