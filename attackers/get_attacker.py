from .black_box import black_box_benchmarks


def get_attacker(cfg):
  if cfg.attacker == 'black_box':
    attacker = black_box_benchmarks
  else:
    raise NotImplementedError

  return attacker
