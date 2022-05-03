from functools import partial
import opacus.data_loader
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DistributedDPOptimizer, DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from opacus.utils.uniform_sampler import UniformWithReplacementSampler, DistributedUniformWithReplacementSampler
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


def clip_and_accumulate(self):
  """ Calculate a layer's grad_sample correctly when the batch dimension is merged with another dimension in the forward
  pass. For example:
   x = x.reshape(batch_size*m, ...)
   y = layer(x)
   y = y.reshape(batch_size, m, ...)
  """
  grad_samples = self.grad_samples
  dims = [grad_sample.shape[0] for grad_sample in grad_samples]
  batch_size = min(dims)
  if any(dim != batch_size for dim in dims):
    quotients = [dim//batch_size for dim in dims]
    remainders = [dim%batch_size for dim in dims]
    assert all(remainder == 0 for remainder in remainders), 'Incorrect batch size.'
    for i, quotient in enumerate(quotients):
      if quotient > 1:
        grad_samples[i] = torch.sum(grad_samples[i].view(batch_size, quotient, *grad_samples[i].shape[1:]), dim=1)

  per_param_norms = [
    g.view(len(g), -1).norm(2, dim=-1) for g in grad_samples
  ]
  per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
  per_sample_clip_factor = (self.max_grad_norm/(per_sample_norms+1e-6)).clamp(
    max=1.0
  )

  for p, grad_sample in zip(self.params, grad_samples):
    _check_processed_flag(p.grad_sample)

    grad = torch.einsum('i,i...', per_sample_clip_factor, grad_sample)

    if p.summed_grad is not None:
      p.summed_grad += grad
    else:
      p.summed_grad = grad

    _mark_as_processed(p.grad_sample)


def __iter_sampler__(self):
  num_batches = int(1/self.sample_rate)
  while num_batches > 0:
    mask = (
        torch.rand(self.num_samples, generator=self.generator)
        < self.sample_rate
    )
    indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
    if len(indices) > 0:
      yield indices
      num_batches -= 1


def __iter_ddp_sampler__(self):
  if self.shuffle:
    g = torch.Generator()
    g.manual_seed(self.shuffle_seed+self.epoch)
    indices = torch.randperm(self.total_size, generator=g)  # type: ignore
  else:
    indices = torch.arange(self.total_size)  # type: ignore

  indices = indices[self.rank:self.total_size:self.num_replicas]
  assert len(indices) == self.num_samples

  num_batches = self.num_batches
  while num_batches > 0:
    mask = (
        torch.rand(self.num_samples, generator=self.generator)
        < self.sample_rate
    )
    selected_examples = mask.nonzero(as_tuple=False).reshape(-1)
    if len(selected_examples) > 0:
      yield indices[selected_examples]
      num_batches -= 1


def _collate(batch, collate_fn, sample_empty_shapes):
  if len(batch) > 0:
    return collate_fn(batch)
  else:
    return [torch.zeros(x) for x in sample_empty_shapes]


def wrap_collate_with_empty(collate_fn, sample_empty_shapes):
  return partial(_collate, collate_fn=collate_fn, sample_empty_shapes=sample_empty_shapes)


def patch_opacus():
  # make closure compatible with lightning
  DistributedDPOptimizer.step = step

  # make number of steps per epoch consistent with PyTorch DDP
  DPDataLoader.from_data_loader = from_data_loader

  # calculate grad_sample correctly when the batch dimension is merged with another dimension in the forward pass
  DPOptimizer.clip_and_accumulate = clip_and_accumulate

  # sampler handles empty batch
  UniformWithReplacementSampler.__iter__ = __iter_sampler__
  DistributedUniformWithReplacementSampler.__iter__ = __iter_ddp_sampler__

  # make it pickle-able
  opacus.data_loader.wrap_collate_with_empty = wrap_collate_with_empty
