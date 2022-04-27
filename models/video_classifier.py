import torch

from .base_classifier import BaseClassifierModule


class VideoClassifierModule(BaseClassifierModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.ensemble_method = 'sum'  # sum or max
    self._prepare_ensemble()

  def training_step(self, batch, batch_idx):
    # TODO: rename y_hat to logits and pred to y_hat
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    y_hat = self(x)

    loss = self.get_loss(y_hat, y)
    self.log('train/loss', loss, batch_size=batch_size, prog_bar=True)

    pred = self.get_pred(y_hat)
    for name, get_stat in self.metrics.items():
      self.log(f'train/{name}', get_stat(pred, y), batch_size=batch_size, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    y_hat = self(x)

    loss = self.get_loss(y_hat, y)
    self.log('val/loss', loss, batch_size=batch_size, sync_dist=True, prog_bar=True)

    pred = self.get_pred(y_hat)
    #for name, get_stat in self.metrics.items():
    #  self.log(f'val/{name}', get_stat(pred, y), batch_size=batch_size, sync_dist=True, prog_bar=True)

    video_ids = batch['video_index'].clone()
    self._ensemble_at_video_level(pred, y, video_ids)

  def test_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    y_hat = self(x)

    loss = self.get_loss(y_hat, y)
    self.log('test/loss', loss, batch_size=batch_size, sync_dist=True, prog_bar=True)

    pred = self.get_pred(y_hat)
    video_ids = batch['video_index'].clone()
    self._ensemble_at_video_level(pred, y, video_ids)

  def predict_step(self, batch, batch_idx):
    pass

  def on_validation_epoch_end(self):
    self._collect_results('val')
    self._prepare_ensemble()

  def on_test_epoch_end(self):
    self._collect_results('test')
    self._prepare_ensemble()

  def _prepare_ensemble(self):
    # These are used for data ensembling in the test stage.
    self.video_preds = {}
    self.video_labels = {}
    self.video_clips_cnts = {}

  def _collect_results(self, phase='val'):
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
      self.log(f'{phase}/{name}', get_stat(video_preds, video_labels), on_epoch=True, prog_bar=True)

  def _ensemble_at_video_level(self, preds, labels, video_ids):
    for i in range(preds.shape[0]):
      vid_id = int(video_ids[i])
      self.video_labels[vid_id] = labels[i]
      if vid_id not in self.video_preds:
        self.video_preds[vid_id] = torch.zeros(
          (self.cfg.num_classes), device=preds.device, dtype=preds.dtype
        )
        self.video_clips_cnts[vid_id] = 0

      if self.ensemble_method == 'sum':
        self.video_preds[vid_id] += preds[i]
      elif self.ensemble_method == 'max':
        self.video_preds[vid_id] = torch.max(self.video_preds[vid_id], preds[i])
      self.video_clips_cnts[vid_id] += 1
