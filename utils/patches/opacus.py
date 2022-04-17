from opacus.data_loader import DPDataLoader
from opacus.optimizers import DistributedDPOptimizer, DPOptimizer
from opacus.optimizers.optimizer import _get_flat_grad_sample
import torch
from torch.utils.data import IterableDataset


@classmethod
def from_data_loader(cls, data_loader, *, distributed=False, generator=None):
  if isinstance(data_loader.dataset, IterableDataset):
    raise ValueError('Uniform sampling is not supported for IterableDataset')

  world_size = torch.distributed.get_world_size() if distributed else 1
  sample_rate = world_size/len(data_loader)
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


def step(self, closure=None):
  if closure is not None:
    with torch.enable_grad():
      closure()

  if self.pre_step():
    self.reduce_gradients()
    return self.original_optimizer.step()
  else:
    return None


@property
def grad_samples(self):
  ret, dims = [], []
  for p in self.params:
    grad_sample = _get_flat_grad_sample(p)
    ret.append(grad_sample)
    dims.append(grad_sample.shape[0])

  """ Calculate a layer's grad_sample correctly when the batch dimension is merged with another dimension in the forward 
  pass. For example:
   x = x.reshape(batch_size*m, ...)
   y = layer(x)
   y = y.reshape(batch_size, m, ...)
  """
  batch_size = min(dims)
  if any(dim != batch_size for dim in dims):
    quotients = [dim//batch_size for dim in dims]
    remainders = [dim%batch_size for dim in dims]
    assert all(remainder == 0 for remainder in remainders), 'Incorrect batch size.'
    for i, quotient in enumerate(quotients):
      if quotient > 1:
        ret[i] = torch.sum(ret[i].view(batch_size, quotient, *ret[i].shape[1:]), dim=1)

  return ret


def patch_opacus():
  # make closure compatible with lightning
  DistributedDPOptimizer.step = step

  # make number of steps per epoch consistent with PyTorch DDP
  DPDataLoader.from_data_loader = from_data_loader

  # calculate grad_sample correctly when the batch dimension is merged with another dimension in the forward pass
  DPOptimizer.grad_samples = grad_samples