def get_name(cfg):
  name = f'{cfg.dataset}_{cfg.net}_{cfg.mode}_epoch{cfg.num_epochs}_bs{cfg.batch_size}_lr{cfg.lr}'

  if cfg.dp:
    name += f'_sigma{cfg.sigma}_c{cfg.c}'

  if cfg.phase == 'tune':
    name += '_tune'

  return name
