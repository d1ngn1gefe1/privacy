import os
import torch

from models.modules.base_classifier import BaseClassifierModule
import utils


class VideoClassifierModule(BaseClassifierModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.ensemble_method = 'sum'  # sum or max
    self._prepare_ensemble()

  def training_step(self, batch, batch_idx):
    batch_size = batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('train/loss', loss, prog_bar=True, batch_size=batch_size)

    y_hat = self.get_pred(logits)
    for name, get_stat in self.metrics.items():
      self.log(f'train/{name}', get_stat(y_hat, y), prog_bar=True, batch_size=batch_size)

    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('val/loss', loss, sync_dist=True, prog_bar=True, batch_size=batch_size)

    y_hat = self.get_pred(logits)
    video_ids = batch['video_index'].clone()
    self._ensemble_at_video_level(y_hat, y, video_ids)

  def test_step(self, batch, batch_idx):
    batch_size = batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('test/loss', loss, sync_dist=True, prog_bar=True, batch_size=batch_size)

    y_hat = self.get_pred(logits)
    video_ids = batch['video_index'].clone()
    self._ensemble_at_video_level(y_hat, y, video_ids)

  def predict_step(self, batch, batch_idx):
    pass

  def on_validation_epoch_end(self):
    self._sync_and_aggregate_results('val')
    self._prepare_ensemble()

  def on_test_epoch_end(self):
    self._sync_and_aggregate_results('test')
    self._prepare_ensemble()

  def _prepare_ensemble(self):
    # These are used for data ensembling in the val/test stage.
    self.video_preds = {}
    self.video_labels = {}
    self.video_clips_cnts = {}

  def _sync_and_aggregate_results(self, phase):
    if utils.is_ddp():
      rank = utils.get_rank()

      if rank != 0:
        torch.save(utils.to(self.video_preds, self.cfg.gpus[0]), f'video_preds_{rank}.pt')
        torch.save(utils.to(self.video_labels, self.cfg.gpus[0]), f'video_labels_{rank}.pt')
        torch.save(self.video_clips_cnts, f'video_clips_cnts_{rank}.pt')

      utils.barrier()

      if rank == 0:
        for rank_sub in range(1, len(self.cfg.gpus)):
          video_preds = torch.load(f'video_preds_{rank_sub}.pt')
          video_labels = torch.load(f'video_labels_{rank_sub}.pt')
          video_clips_cnts = torch.load(f'video_clips_cnts_{rank_sub}.pt')
          os.remove(f'video_preds_{rank_sub}.pt')
          os.remove(f'video_labels_{rank_sub}.pt')
          os.remove(f'video_clips_cnts_{rank_sub}.pt')
          self.video_preds.update(video_preds)
          self.video_labels.update(video_labels)
          self.video_clips_cnts.update(video_clips_cnts)

        self._aggregate_results(phase)

    else:
      self._aggregate_results(phase)

  def _aggregate_results(self, phase):
    for video_id in self.video_preds:
      self.video_preds[video_id] = (
        self.video_preds[video_id] / self.video_clips_cnts[video_id]
      )
      video_preds = torch.stack(list(self.video_preds.values()), dim=0)
      video_labels = torch.tensor(
        list(self.video_labels.values()),
        device=self.video_labels[video_id].device,
      )
    for name, get_stat in self.metrics.items():
      stat = get_stat(video_preds, video_labels)
      self.log(f'{phase}/{name}', stat, on_epoch=True, prog_bar=True, rank_zero_only=True)

  def _ensemble_at_video_level(self, preds, labels, video_ids):
    for i in range(preds.shape[0]):
      video_id = int(video_ids[i])
      self.video_labels[video_id] = labels[i]
      if video_id not in self.video_preds:
        self.video_preds[video_id] = torch.zeros(self.cfg.num_classes, device=preds.device, dtype=preds.dtype)
        self.video_clips_cnts[video_id] = 0

      if self.ensemble_method == 'sum':
        self.video_preds[video_id] += preds[i]
      elif self.ensemble_method == 'max':
        self.video_preds[video_id] = torch.max(self.video_preds[video_id], preds[i])
      self.video_clips_cnts[video_id] += 1
