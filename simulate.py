from omegaconf import OmegaConf
from opacus.accountants.analysis import rdp as privacy_analysis

from data import get_data
from models import get_model
import utils


def main():
  cfg = OmegaConf.load('configs/diving48/mvit.yaml')
  utils.setup(cfg, 'simulate')

  data = get_data(cfg)
  data.setup()
  model = get_model(cfg)

  epoch = 0
  epsilon, alpha_best = get_privacy_spent(epoch, data, model, cfg.sigma, cfg.delta, len(cfg.gpus))
  print(f'Epoch {epoch}: epsilon={epsilon}, alpha_best={alpha_best}')


def get_privacy_spent(epoch, data, model, sigma, delta, world_size):
  dataloader = data.train_dataloader()
  alphas = [1+x/10.0 for x in range(1, 100)]+list(range(12, 64))
  sample_rate = world_size/len(dataloader)

  num_steps = (epoch+1)*len(dataloader)/world_size
  rdp = privacy_analysis.compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=num_steps, orders=alphas)
  epsilon, alpha_best = privacy_analysis.get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)

  print(f'sample_rate={sample_rate}, len_dataloader={len(dataloader)}')
  return epsilon, alpha_best


if __name__ == '__main__':
  main()
