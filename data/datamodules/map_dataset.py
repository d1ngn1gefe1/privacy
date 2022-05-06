# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_dataset.py

import logging
from pytorchvideo.data import RandomClipSampler
from pytorchvideo.data.video import VideoPathHandler
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class MapDataset(Dataset):
  _MAX_CONSECUTIVE_FAILURES = 10

  def __init__(self, labeled_videos, clip_sampler, transform, decode_audio=True, decode_video=True, decoder='pyav'):
    super().__init__()

    self._labeled_videos = labeled_videos
    self._clip_sampler = clip_sampler
    self._transform = transform
    self._decode_audio = decode_audio
    self._decode_video = decode_video
    self._decoder = decoder
    self.video_path_handler = VideoPathHandler()

  @classmethod
  def from_iterable_dataset(cls, iterable_dataset):
    labeled_videos = iterable_dataset._labeled_videos
    clip_sampler = iterable_dataset._clip_sampler
    transform = iterable_dataset._transform
    decode_audio = iterable_dataset._decode_audio
    decode_video = iterable_dataset._decode_video
    decoder = iterable_dataset._decoder

    assert isinstance(clip_sampler, RandomClipSampler)

    return cls(labeled_videos, clip_sampler, transform, decode_audio, decode_video, decoder)

  def __len__(self):
    return len(self._labeled_videos)

  def __getitem__(self, video_index):
    for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
      try:
        video_path, info_dict = self._labeled_videos[video_index]
        video = self.video_path_handler.video_from_path(
          video_path,
          decode_audio=self._decode_audio,
          decode_video=self._decode_video,
          decoder=self._decoder,
        )
        _loaded_video_label = (video, info_dict, video_index)
      except Exception as e:
        logger.debug(
          'Failed to load video with error: {}; trial {}'.format(
            e,
            i_try,
          )
        )
        continue

      (
        clip_start,
        clip_end,
        clip_index,
        aug_index,
        is_last_clip,
      ) = self._clip_sampler(None, video.duration, info_dict)

      _loaded_clip = video.get_clip(clip_start, clip_end)

      video_is_null = (
          _loaded_clip is None or _loaded_clip['video'] is None
      )
      if (
          is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
      ) or video_is_null:
        _loaded_video_label[0].close()
        if video_is_null:
          logger.debug(
            'Failed to load clip {}; trial {}'.format(video.name, i_try)
          )
          continue

      frames = _loaded_clip['video']
      audio_samples = _loaded_clip['audio']
      sample_dict = {
        'video': frames,
        'video_name': video.name,
        'video_index': video_index,
        'clip_index': clip_index,
        'aug_index': aug_index,
        **info_dict,
        **({'audio': audio_samples} if audio_samples is not None else {}),
      }
      if self._transform is not None:
        sample_dict = self._transform(sample_dict)

      return sample_dict
    else:
      raise RuntimeError(
        f'Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries.'
      )
