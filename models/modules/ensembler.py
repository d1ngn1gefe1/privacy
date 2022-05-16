import os
import torch

import utils


class Ensembler:
  def __init__(self, num_classes, gpus, ensemble_method='sum'):
    self.num_classes = num_classes
    self.gpus = gpus
    self.ensemble_method = ensemble_method

    self.video_preds = {}
    self.video_labels = {}
    self.video_clips_cnts = {}

  def ensemble_at_video_level(self, preds, labels, video_ids):
    for i in range(preds.shape[0]):
      video_id = int(video_ids[i])
      self.video_labels[video_id] = labels[i]
      if video_id not in self.video_preds:
        self.video_preds[video_id] = torch.zeros(self.num_classes, device=preds.device, dtype=preds.dtype)
        self.video_clips_cnts[video_id] = 0

      if self.ensemble_method == 'sum':
        self.video_preds[video_id] += preds[i]
      elif self.ensemble_method == 'max':
        self.video_preds[video_id] = torch.max(self.video_preds[video_id], preds[i])
      self.video_clips_cnts[video_id] += 1

  def sync_and_aggregate_results(self):
    video_preds, video_labels = None, None

    if utils.is_ddp():
      rank = utils.get_rank()

      if rank != 0:
        torch.save(utils.to(self.video_preds, self.gpus[0]), f'video_preds_{rank}.pt')
        torch.save(utils.to(self.video_labels, self.gpus[0]), f'video_labels_{rank}.pt')
        torch.save(self.video_clips_cnts, f'video_clips_cnts_{rank}.pt')

      utils.barrier()

      if rank == 0:
        for rank_sub in range(1, len(self.gpus)):
          video_preds = torch.load(f'video_preds_{rank_sub}.pt')
          video_labels = torch.load(f'video_labels_{rank_sub}.pt')
          video_clips_cnts = torch.load(f'video_clips_cnts_{rank_sub}.pt')
          os.remove(f'video_preds_{rank_sub}.pt')
          os.remove(f'video_labels_{rank_sub}.pt')
          os.remove(f'video_clips_cnts_{rank_sub}.pt')
          self.video_preds.update(video_preds)
          self.video_labels.update(video_labels)
          self.video_clips_cnts.update(video_clips_cnts)

        video_preds, video_labels = self._aggregate_results()

    else:
      video_preds, video_labels = self._aggregate_results()

    # reset
    self.video_preds = {}
    self.video_labels = {}
    self.video_clips_cnts = {}

    return video_preds, video_labels

  def _aggregate_results(self):
    for video_id in self.video_preds:
      self.video_preds[video_id] = self.video_preds[video_id]/self.video_clips_cnts[video_id]

    video_preds = torch.stack(list(self.video_preds.values()), dim=0)
    video_labels = torch.tensor(list(self.video_labels.values()), device=video_preds.device)

    return video_preds, video_labels
