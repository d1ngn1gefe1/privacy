import numpy as np
from omegaconf import OmegaConf
import os

from attackers import get_attacker
import utils


def main():
  cfg = OmegaConf.load('configs/cifar100/opacus_net.yaml')
  utils.setup(cfg, 'attach')

  if cfg.attacker == 'black_box':
    path_result = os.path.join(cfg.dir_weights, cfg.relpath_predict)+'.npz'
    result = np.load(path_result)
    attacker_cls = get_attacker(cfg)

    train_results = (result['train_preds'], result['train_gts'])
    test_results = (result['test_preds'], result['test_gts'])
    num_classes = result['train_preds'].shape[1]
    attacker = attacker_cls(train_results, test_results,
      train_results, test_results, num_classes)
    attacker.attack()
  else:
    raise NotImplementedError


if __name__ == '__main__':
  main()
