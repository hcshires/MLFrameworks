import torch

class GPUDS(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.data = [(torch.tensor(x).to(device), torch.tensor(y).to(device)) for x, y, in dataset]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]