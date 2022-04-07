from pytorch_lightning import LightningModule

from .nets import get_net


class VideoClassifierModule(LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.net = get_net(cfg.net, cfg.num_classes, cfg.mode != 'from_scratch', cfg.dir_weights)

    # set trainable parameters
    if cfg.mode == 'linear_probing' or cfg.mode == 'adapter':
      for param in self.net.parameters():
        param.requires_grad = False
      for param in self.net.get_classifier().parameters():
        param.requires_grad = True

  def forward(self, x):
    return self.net(x)

  def training_step(self, batch, batch_idx):
    pass

  def validation_step(self, batch, batch_idx):
    pass

  def test_step(self, batch, batch_idx):
    pass

  def predict_step(self, batch, batch_idx):
    pass

  def configure_optimizers(self):
    pass