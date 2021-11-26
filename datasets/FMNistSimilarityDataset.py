from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torch


class FMNistSimilarityDataset:

    def __init__(self, dataset_path, transform, samples_per_class, n_labels):
        super(FMNistSimilarityDataset, self).__init__()

        self.new_dataset = []
        self.samples_per_class = samples_per_class
        self.n_labels = n_labels

        def get_samples(labels):
            sampled = []
            for i in range(self.n_labels):
                indices = torch.nonzero(labels == i).view(-1)
                random_indices = torch.randperm(indices.shape[0])[:self.samples_per_class]
                sampled.append(indices[random_indices])

            return torch.hstack(sampled)

        dataset = datasets.FashionMNIST(
            dataset_path,
            transform=transforms.Compose(transform),
            download=True,
            train=False
        )

        samples = get_samples(dataset.targets)
        self.dataset = Subset(dataset, samples)
        self.len = len(self.dataset)

    def __len__(self):
        return self.len

    def get_dataset(self):
        return self.dataset
