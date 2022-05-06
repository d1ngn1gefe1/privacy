from models.modules.base_classifier import BaseClassifierModule


class ImageClassifierModule(BaseClassifierModule):
  def __init__(self, cfg):
    super().__init__(cfg)

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('train/loss', loss, prog_bar=True)

    y_hat = self.get_pred(logits)
    for name, get_stat in self.metrics.items():
      self.log(f'train/{name}', get_stat(y_hat, y), prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('val/loss', loss, sync_dist=True, prog_bar=True)

    y_hat = self.get_pred(logits)
    for name, get_stat in self.metrics.items():
      stat = get_stat(y_hat, y)
      self.log(f'val/{name}', stat, sync_dist=True, prog_bar=True)

  def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('test/loss', loss, sync_dist=True, prog_bar=True)

    y_hat = self.get_pred(logits)
    for name, get_stat in self.metrics.items():
      stat = get_stat(y_hat, y)
      self.log(f'test/{name}', stat, sync_dist=True, prog_bar=True)

  def predict_step(self, batch, batch_idx):
    pass
