from models.modules.base_classifier import BaseClassifierModule
from .misc import handle_multi_view


class ImageClassifierModule(BaseClassifierModule):
  def __init__(self, cfg):
    super().__init__(cfg)

  def training_step(self, batch, batch_idx):
    x, y = batch
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
    x, y = batch
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('val/loss', loss, sync_dist=True, prog_bar=True)

    y_hat = self.get_pred(logits)
    for name, metric in self.metrics_val.items():
      result = metric(y_hat, y)
      self.log(f'val/{name}', result, sync_dist=True, prog_bar=True)

  def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('test/loss', loss, sync_dist=True, prog_bar=True)

    y_hat = self.get_pred(logits)
    for name, metric in self.metrics_test.items():
      result = metric(y_hat, y)
      self.log(f'test/{name}', result, sync_dist=True, prog_bar=True)

  def predict_step(self, batch, batch_idx):
    pass
