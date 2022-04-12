from opacus.optimizers import DistributedDPOptimizer
import torch


def step(self, closure=None):
  if closure is not None:
    with torch.enable_grad():
      closure()

  if self.pre_step():
    self.reduce_gradients()
    return self.original_optimizer.step(closure)
  else:
    return None


# fix bugs in opacus
def patch_opacus():
  DistributedDPOptimizer.step = step
