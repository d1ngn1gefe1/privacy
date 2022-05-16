from functools import partial
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


def multilabel_loss_with_uncertainty(logits, labels, uncertainty_approach='u_zeros'):
  if uncertainty_approach == 'u_zeros':
    labels[labels == -1] = 0
  elif uncertainty_approach == 'u_ones':
    labels[labels == -1] = 1
  elif uncertainty_approach == 'ignore':
    mask = labels != -1
    logits = logits[mask]
    labels = labels[mask]

  labels = labels.int()
  loss = F.multilabel_soft_margin_loss(logits, labels)
  return loss


class MaskedAUROC(nn.Module):
  """ AUROC with handle for uncertain labels """
  def __init__(self, num_classes=5):
    super().__init__()
    self.num_classes = num_classes
    self.meters = {k: torchmetrics.AUROC(pos_label=1) for k in range(num_classes)}

  def forward(self, logits, labels):
    stats = []
    for cls, meter in self.meters.items():
      curr_logits = logits[:, cls]
      curr_labels = labels[:, cls]

      mask = curr_labels != -1
      if mask.sum() == 0:
        continue
      valid_logits = curr_logits[mask]
      valid_labels = curr_labels[mask]

      stat = meter(valid_logits, valid_labels) 
      stats.append(stat)

    return sum(stats)/len(stats)


def handle_multi_view(x, y):
  if isinstance(x, list):
    num_repeats = len(x)
    batch_size = x[0].shape[0]
    x = torch.cat(x)
    y = y.repeat(num_repeats)
  else:
    batch_size = x.shape[0]

  return x, y, batch_size


def set_mode(net, cfg):
  for param in net.parameters():
    param.requires_grad = False

  if cfg.mode == 'from_scratch' or cfg.mode == 'full_tuning':
    for param in net.parameters():
      param.requires_grad = True

  elif cfg.mode == 'linear_probing' or cfg.mode == 'adapter':
    for param in net.get_classifier().parameters():
      param.requires_grad = True

  elif cfg.mode == 'sparse_tuning':
    for param in net.get_classifier().parameters():
      param.requires_grad = True
    for param in itertools.chain(*[norm.parameters() for norm in net.get_norms()]):
      param.requires_grad = True

  else:
    raise NotImplementedError


def set_task(cfg):
  if cfg.task == 'multi-class':
    get_loss = F.cross_entropy
    get_pred = partial(F.softmax, dim=-1)  # avoid using lambda because it cannot be pickled
    metrics = nn.ModuleDict({'acc1': torchmetrics.Accuracy(average='micro', top_k=1),
                             'acc5': torchmetrics.Accuracy(average='micro', top_k=5)})

  elif cfg.task == 'multi-label':
    if hasattr(cfg, 'uncertainty'):
      get_loss = partial(multilabel_loss_with_uncertainty, uncertainty_approach=cfg.uncertainty)
      metrics = nn.ModuleDict({'roc': MaskedAUROC(num_classes=cfg.num_classes)})
    else:
      get_loss = F.multilabel_soft_margin_loss
      metrics = nn.ModuleDict({'acc': torchmetrics.Accuracy(average='macro', num_classes=cfg.num_classes),
                               'roc': torchmetrics.AUROC(num_classes=cfg.num_classes)})
    get_pred = torch.sigmoid

  else:
    raise NotImplementedError

  return get_loss, get_pred, metrics
