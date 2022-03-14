import math
from opacus import PrivacyEngine
import torch
from types import MethodType

from .image_classifier import ImageClassifierModule


def on_train_epoch_end(self):
  epsilon = self.privacy_engine.get_epsilon(self.cfg.delta)
  self.log('epsilon', epsilon, on_epoch=True, sync_dist=True, prog_bar=True)


def configure_optimizers(self):
  dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()
  sample_rate = 1/len(dataloader)
  expected_batch_size = int(len(dataloader.dataset)*sample_rate)
  distributed = len(self.cfg.gpus) > 1
  if distributed:
    world_size = torch.distributed.get_world_size()
    expected_batch_size /= world_size  # expected_batch_size is the per-worker batch size
    num_layers = len([(n, p) for n, p in self.named_parameters() if p.requires_grad])
    max_grad_norm = [self.cfg.c/math.sqrt(num_layers)]*num_layers
    clipping = 'per_layer'
  else:
    max_grad_norm = self.cfg.c
    clipping = 'flat'

  optimizer = self.privacy_engine._prepare_optimizer(self.configure_optimizers_old(),
                                                     noise_multiplier=self.cfg.sigma,
                                                     max_grad_norm=max_grad_norm,
                                                     expected_batch_size=expected_batch_size,
                                                     distributed=distributed,
                                                     clipping=clipping)
  optimizer.attach_step_hook(self.privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))
  return optimizer


def get_model(cfg):
  model = ImageClassifierModule(cfg)

  if cfg.dp:
    model.privacy_engine = PrivacyEngine()
    model.on_train_epoch_end = MethodType(on_train_epoch_end, model)

    # Reference: https://github.com/pytorch/opacus/blob/main/opacus/privacy_engine.py#L153
    model.net = model.privacy_engine._prepare_model(model.net)
    # model.net.register_forward_pre_hook(forbid_accumulation_hook)  # todo: lightning will trigger an error

    # Reference: https://github.com/pytorch/opacus/blob/main/opacus/privacy_engine.py#L120
    model.configure_optimizers_old = model.configure_optimizers
    model.configure_optimizers = MethodType(configure_optimizers, model)

  return model
