from torch.utils.data import Dataset
import torch


class LatentVectorsDataset(Dataset):

    def __init__(self, dataset_path):
        super(LatentVectorsDataset, self).__init__()

        self.data = torch.load(dataset_path)
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index >= self.len:
            raise IndexError(f"Index {index} is out of bounds for array of size {self.len}")

        latent_vector = self.data[index, :]
        return latent_vector
