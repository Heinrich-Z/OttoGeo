from torchvision import transforms
from torch.utils.data import Dataset
import h5py
import torch

class CustomDataset(Dataset):
    def __init__(self, file_path, field_names, case_name):
        self.file_path = file_path
        self.file_object = h5py.File(file_path, 'r')
        self.field_names = field_names
        self.case_name = case_name
        self.transform = None

        with h5py.File(self.file_path, 'r') as f:
            self.inputs = f[self.field_names[0]][:]
            self.output = f[self.field_names[1]][:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.output[idx], dtype=torch.float32)
        sample = (x, y)
        return self.transform(sample)

class DataTransform(object):

    def __init__(self, mean, std, case_name):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.case_name = case_name

    def __call__(self, sample):
        x, y = sample
        x = (x - self.mean) / self.std

        if self.case_name == 'ViT':
            resize = transforms.Resize((224, 224))
            x = resize(x)
        
        return x, y
