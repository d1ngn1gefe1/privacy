# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/train.py#L341

import itertools
import torch


class MapStyleDataset(torch.utils.data.Dataset):
  """
  To ensure a constant number of samples are retrieved from the dataset we use this
  LimitDataset wrapper. This is necessary because several of the underlying videos
  may be corrupted while fetching or decoding, however, we always want the same
  number of steps per epoch.
  """

  def __init__(self, dataset):
    super().__init__()
    self.dataset = dataset
    self.dataset_iter = itertools.chain.from_iterable(
      itertools.repeat(iter(dataset), 2)
    )

  def __getitem__(self, index):
    return next(self.dataset_iter)

  def __len__(self):
    return self.dataset.num_videos
