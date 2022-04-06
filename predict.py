import numpy as np
from omegaconf import OmegaConf
import os
import torch
from tqdm import tqdm

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def inference(model, dataloader):
  preds = []
  gts = []
  for batch in tqdm(dataloader):
    x, y = batch
    with torch.no_grad():
      y_hat = model(x)
      y_hat = model.get_pred(y_hat)
    preds.append(y_hat.cpu().numpy())
    gts.append(y.cpu().numpy())

  preds = np.concatenate(preds, 0)
  gts = np.concatenate(gts)
  save_dict = {'preds': preds, 'gts': gts}
  return save_dict


def main():
  #cfg = OmegaConf.load('configs/cifar100/opacus_net.yaml')
  cfg = OmegaConf.load('configs/cifar100/resnet.yaml')
  cfg.phase = 'test'
  cfg.name = utils.get_name(cfg)

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  path_ckpt = os.path.join(cfg.dir_weights, cfg.relpath_ckpt)
  trainer.predict(model, datamodule=data, ckpt_path=path_ckpt)

  # Using python runtime to do inference
  model.eval()
  predict_dataloader = data.predict_dataloader()
  predict_save_dict = inference(model, predict_dataloader)
  train_dataloader = data.train_dataloader()
  train_save_dict = inference(model, train_dataloader)

  path_predict = os.path.join(cfg.dir_weights, cfg.relpath_predict)
  np.savez(path_predict, train_preds=train_save_dict['preds'],
    train_gts=train_save_dict['gts'], test_preds=predict_save_dict['preds'],
    test_gts=predict_save_dict['gts'])

if __name__ == '__main__':
  main()

