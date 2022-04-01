import inspect
import math
from opacus import PrivacyEngine
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
import torch
from types import MethodType


def on_train_epoch_end(self):
  epsilon = self.privacy_engine.get_epsilon(self.cfg.delta)
  self.log('epsilon', epsilon, on_epoch=True, sync_dist=False, prog_bar=True)  # same across all devices, no sync

  dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()
  sample_rate = 1/len(dataloader)
  alphas = [1+x/10.0 for x in range(1, 100)]+list(range(12, 64))
  rdp = compute_rdp(q=sample_rate, noise_multiplier=self.cfg.sigma,
                    steps=self.trainer.current_epoch*math.ceil(1/sample_rate),
                    orders=alphas)
  epsilon2, _ = get_privacy_spent(orders=alphas, rdp=rdp, delta=self.cfg.delta)
  self.log('epsilon2', epsilon2, on_epoch=True, sync_dist=False, prog_bar=True)

def configure_optimizers(self):
  dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()
  sample_rate = 1/len(dataloader)
  expected_batch_size = int(len(dataloader.dataset)*sample_rate)
  # todo: verify ddp batch size
  # print(f'{sample_rate}, {len(dataloader)}, {len(dataloader.dataset)}')

  if len(self.cfg.gpus) > 1:
    distributed = True
    world_size = torch.distributed.get_world_size()
    expected_batch_size /= world_size  # expected_batch_size is the per-worker batch size
    num_layers = len([(n, p) for n, p in self.named_parameters() if p.requires_grad])
    max_grad_norm = [self.cfg.c/math.sqrt(num_layers)]*num_layers
    clipping = 'per_layer'  # preferred when using ddp
  else:
    distributed = False
    max_grad_norm = self.cfg.c
    clipping = 'flat'

  dict_optimizers = self.configure_optimizers_old()
  optimizer_old, scheduler_old = dict_optimizers['optimizer'], dict_optimizers['lr_scheduler']

  # optimizer
  optimizer = self.privacy_engine._prepare_optimizer(optimizer_old,
                                                     distributed=distributed,
                                                     noise_multiplier=self.cfg.sigma,
                                                     max_grad_norm=max_grad_norm,
                                                     expected_batch_size=expected_batch_size,
                                                     clipping=clipping)
  optimizer.attach_step_hook(self.privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))

  # lr scheduler
  kwargs = {key:scheduler_old.__dict__[key]
            for key in inspect.signature(scheduler_old.__class__.__init__).parameters.keys()
            if key not in ['self', 'optimizer']}
  scheduler = scheduler_old.__class__(optimizer, **kwargs)

  print(f'Net: {type(self.net)}, Optimizer: {type(optimizer)}')
  return {'optimizer': optimizer, 'lr_scheduler': scheduler}


# Reference: https://github.com/pytorch/opacus/blob/main/examples/mnist_lightning.py
def make_private(model):
  model.privacy_engine = PrivacyEngine()
  model.on_train_epoch_end = MethodType(on_train_epoch_end, model)

  # Reference: https://github.com/pytorch/opacus/blob/main/opacus/privacy_engine.py#L153
  model.net = model.privacy_engine._prepare_model(model.net)
  model.net.get_classifier = model.net._module.get_classifier
  # model.net.register_forward_pre_hook(forbid_accumulation_hook)  # todo: lightning will trigger an error

  # Reference: https://github.com/pytorch/opacus/blob/main/opacus/privacy_engine.py#L120
  model.configure_optimizers_old = model.configure_optimizers
  model.configure_optimizers = MethodType(configure_optimizers, model)

  return model
