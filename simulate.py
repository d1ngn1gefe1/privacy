from omegaconf import OmegaConf
from opacus.accountants.analysis import rdp as privacy_analysis

from data import get_data
from models import get_model


def main():
  cfg = OmegaConf.load('configs/cifar100/opacus_net.yaml')

  data = get_data(cfg)
  model = get_model(cfg)
  data.setup()

  epoch = 10
  epsilon, alpha_best = get_privacy_spent(epoch, data, model, cfg.sigma, cfg.delta)
  print(f'Epoch {epoch}: epsilon={epsilon}, alpha_best={alpha_best}')


def get_privacy_spent(epoch, data, model, sigma, delta):
  dataloader = data.train_dataloader()
  alphas = model.privacy_engine.accountant.DEFAULT_ALPHAS
  sample_rate = 1/len(dataloader)

  num_steps = (epoch+1)*len(dataloader)
  rdp = privacy_analysis.compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=num_steps, orders=alphas)
  epsilon, alpha_best = privacy_analysis.get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)
  return epsilon, alpha_best


if __name__ == '__main__':
  main()
