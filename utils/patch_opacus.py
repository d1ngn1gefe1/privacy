from opacus.data_loader import DPDataLoader
from opacus.optimizers import DistributedDPOptimizer
import torch
from torch.utils.data import IterableDataset


def step(self, closure=None):
  if closure is not None:
    with torch.enable_grad():
      closure()

  if self.pre_step():
    self.reduce_gradients()
    return self.original_optimizer.step(closure)
  else:
    return None


@classmethod
def from_data_loader(cls, data_loader, *, distributed=False, generator=None):
  if isinstance(data_loader.dataset, IterableDataset):
    raise ValueError('Uniform sampling is not supported for IterableDataset')

  world_size = torch.distributed.get_world_size() if distributed else 1
  sample_rate = world_size/len(data_loader)
  print(f'from_data_loader: {sample_rate}, {len(data_loader)}')
  return cls(
    dataset=data_loader.dataset,
    sample_rate=sample_rate,
    num_workers=data_loader.num_workers,
    collate_fn=data_loader.collate_fn,
    pin_memory=data_loader.pin_memory,
    drop_last=data_loader.drop_last,
    timeout=data_loader.timeout,
    worker_init_fn=data_loader.worker_init_fn,
    multiprocessing_context=data_loader.multiprocessing_context,
    generator=generator if generator else data_loader.generator,
    prefetch_factor=data_loader.prefetch_factor,
    persistent_workers=data_loader.persistent_workers,
    distributed=distributed,
  )


def patch_opacus():
  # fix bug
  DistributedDPOptimizer.step = step

  # make number of steps per epoch consistent with PyTorch DDP
  DPDataLoader.from_data_loader = from_data_loader