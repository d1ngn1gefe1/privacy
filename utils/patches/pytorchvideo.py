import pytorchvideo.transforms.functional
from pytorchvideo.transforms.functional import _get_param_spatial_crop
import torch
from typing import Tuple


def random_resized_crop(
    frames: torch.Tensor,
    target_height: int,
    target_width: int,
    scale: Tuple[float, float],
    aspect_ratio: Tuple[float, float],
    shift: bool = False,
    log_uniform_ratio: bool = True,
    interpolation: str = 'bilinear',
    num_tries: int = 10,
) -> torch.Tensor:
  assert (
      scale[0] > 0 and scale[1] > 0
  ), 'min and max of scale range must be greater than 0'
  assert (
      aspect_ratio[0] > 0 and aspect_ratio[1] > 0
  ), 'min and max of aspect_ratio range must be greater than 0'

  channels = frames.shape[0]
  t = frames.shape[1]
  height = frames.shape[2]
  width = frames.shape[3]

  i, j, h, w = _get_param_spatial_crop(
    scale, aspect_ratio, height, width, log_uniform_ratio, num_tries
  )

  if not shift:
    cropped = frames[:, :, i: i+h, j: j+w]
    return torch.nn.functional.interpolate(
      cropped,
      size=(target_height, target_width),
      mode=interpolation,
      align_corners=False
    )

  i_, j_, h_, w_ = _get_param_spatial_crop(
    scale, aspect_ratio, height, width, log_uniform_ratio, num_tries
  )
  i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
  j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
  h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
  w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
  cropped = torch.zeros((channels, t, target_height, target_width))
  for ind in range(t):
    cropped[:, ind: ind+1, :, :] = torch.nn.functional.interpolate(
      frames[
      :,
      ind: ind+1,
      i_s[ind]: i_s[ind]+h_s[ind],
      j_s[ind]: j_s[ind]+w_s[ind],
      ],
      size=(target_height, target_width),
      mode=interpolation,
      align_corners=False
    )
  return cropped


def patch_pytorchvideo():
  # suppress the warning in torch.nn.functional.interpolate() by setting align_corners=False
  pytorchvideo.transforms.functional.random_resized_crop = random_resized_crop
