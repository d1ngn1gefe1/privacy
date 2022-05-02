import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


def multilabel_loss_with_uncertainty(
  logits,
  labels,
  uncertainty_approach='u_zeros'
):
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
