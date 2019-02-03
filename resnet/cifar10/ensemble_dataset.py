from torch.utils.data import Dataset

'''
Following https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/2 .
'''
class EnsembleDataset(Dataset):
    def __init__(self, datasets):
         self.datasets = datasets
         
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
         
    def __len__(self):
        return min(len(d) for d in self.datasets)
