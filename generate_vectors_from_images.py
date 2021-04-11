import argparse
import torch
from torch import nn
import numpy as np
import os
from networks import Critic, InverseGenerator
import utils

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--data_path', default='data/mnist/', type=str, help='Directory of the dataset')
parser.add_argument('--pretrained_models_path', default='pretrained_models',
                    type=str, help='Directory for trained models')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--gan_type', default='dcgan', type=str, choices=['wgan', 'dcgan'], help="dcgan or wgan")
parser.add_argument('--output_dim', default=100, type=int, help='dimensions of output feature vector')
parser.add_argument('--generate_from', default='inv_g', type=str, choices=['inv_g', 'd'],
                    help='Generate vectors from images using inverse generator or discriminator (inv_g or d)')
args = parser.parse_args()


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

IMG_SIZE = 32
DIM = 64  # Model dimensionality
NEW_DATASET_PATH = os.path.join(args.data_path, f"FVecFashionMNIST_{args.generate_from.upper()}", args.gan_type)
MODEL_PATH = os.path.join(args.pretrained_models_path, args.gan_type, f"{args.gan_type}_{args.generate_from}.pkl")


def generate_feature_vectors(model: nn.Module, train=True):
    dataloader = utils.get_dataloader(args.data_path, IMG_SIZE, args.batch_size, train)
    feature_vectors, labels = [], []
    for i, data in enumerate(dataloader):
        images, label = data
        feature_vector = model(images.to(device))
        if args.generate_from == 'd':
            feature_vector = feature_vector[1]
        feature_vectors.append(feature_vector.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
    return np.vstack(feature_vectors), np.hstack(labels)


if args.generate_from == 'inv_g':
    model = InverseGenerator(args.output_dim, DIM).to(device)
else:
    model = Critic(args.gan_type, DIM, args.output_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
train_feature_vectors, train_labels = generate_feature_vectors(model)
test_feature_vectors, test_labels = generate_feature_vectors(model, False)
os.makedirs(NEW_DATASET_PATH, exist_ok=True)
new_train_dataset = {
    'data': train_feature_vectors,
    'labels': train_labels
}
new_test_dataset = {
    'data': test_feature_vectors,
    'labels': test_labels
}
torch.save(new_train_dataset, os.path.join(NEW_DATASET_PATH, "training.pt"))
torch.save(new_train_dataset, os.path.join(NEW_DATASET_PATH, "test.pt"))
