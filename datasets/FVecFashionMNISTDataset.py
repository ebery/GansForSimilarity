from torch.utils.data import Dataset
import torch


class FVecFashionMNISTDataset(Dataset):

    def __init__(self, dataset_path):
        super(FVecFashionMNISTDataset, self).__init__()

        dataset = torch.load(dataset_path)
        self.feature_vectors, self.labels = dataset['data'], dataset['labels']
        self.len, self.feature_dim = self.feature_vectors.shape

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index >= self.len:
            raise IndexError(f"Index {index} is out of bounds for array of size {self.len}")

        feature_vector = self.feature_vectors[index, :]
        label = self.labels[index]
        return feature_vector, label
