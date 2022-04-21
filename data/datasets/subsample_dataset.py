from torch.utils.data import Dataset
from typing import Union


class SubsampleDataset(Dataset):
  def __init__(self, dataset, size: Union[int, float]):
    self.dataset = dataset
    self.length = size if isinstance(size, int) else int(size*len(self.dataset))

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    i = int(index*len(self.dataset)/self.length)
    return self.dataset.__getitem__(i)
