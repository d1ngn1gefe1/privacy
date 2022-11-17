from models.modules.base_classifier import BaseClassifierModule
from .ensembler import Ensembler
from .misc import handle_multi_view
import utils


class VideoClassifierModule(BaseClassifierModule):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.ensembler = Ensembler(cfg.num_classes, cfg.gpus)

  def training_step(self, batch, batch_idx):
    x, y = batch['video'], batch['label']
    x, y, batch_size = handle_multi_view(x, y)
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('train/loss', loss, prog_bar=True, batch_size=batch_size)

    y_hat = self.get_pred(logits)
    for name, metric in self.metrics_train.items():
      result = metric(y_hat, y)
      self.log(f'train/{name}', result, prog_bar=True, batch_size=batch_size)

    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('val/loss', loss, sync_dist=True, prog_bar=True, batch_size=batch_size)

    y_hat = self.get_pred(logits)
    video_ids = batch['video_index'].clone()
    self.ensembler.ensemble_at_video_level(y_hat, y, video_ids)

  def test_step(self, batch, batch_idx):
    batch_size = batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('test/loss', loss, sync_dist=True, prog_bar=True, batch_size=batch_size)

    y_hat = self.get_pred(logits)
    video_ids = batch['video_index'].clone()
    self.ensembler.ensemble_at_video_level(y_hat, y, video_ids)

  def predict_step(self, batch, batch_idx):
    pass

  def on_validation_epoch_end(self):
    y_hat, y = self.ensembler.sync_and_aggregate_results()
    if not utils.is_ddp() or utils.get_rank() == 0:
      for name, metric in self.metrics_val.items():
        result = metric(y_hat, y)
        self.log(f'val/{name}', result, on_epoch=True, prog_bar=True, rank_zero_only=True)

  def on_test_epoch_end(self):
    y_hat, y = self.ensembler.sync_and_aggregate_results()
    if not utils.is_ddp() or utils.get_rank() == 0:
      for name, metric in self.metrics_test.items():
        result = metric(y_hat, y)
        self.log(f'test/{name}', result, on_epoch=True, prog_bar=True, rank_zero_only=True)
