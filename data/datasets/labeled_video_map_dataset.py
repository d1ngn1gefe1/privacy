import torch


class LabeledVideoMapDataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    super().__init__()
    self.dataset = dataset

  def __len__(self):
    return self.dataset.num_videos

  def __getitem__(self, index):
    for i_try in range(self.dataset._MAX_CONSECUTIVE_FAILURES):
      video_index = index
      try:
        video_path, info_dict = self.dataset._labeled_videos[video_index]
        video = self.dataset.video_path_handler.video_from_path(
          video_path,
          decode_audio=self.dataset._decode_audio,
          decode_video=self.dataset._decode_video,
          decoder=self.dataset._decoder,
        )
        self.dataset._loaded_video_label = (video, info_dict, video_index)
      except Exception as e:
        print(f'Failed to load video with error: {e}; trial {i_try}')
        continue

      (
        clip_start,
        clip_end,
        clip_index,
        aug_index,
        is_last_clip,
      ) = self.dataset._clip_sampler(self.dataset._last_clip_end_time, video.duration, info_dict)

      if isinstance(clip_start, list):  # multi-clip
        if aug_index[0] == 0:
          self.dataset._loaded_clip = {}
          loaded_clip_list = []
          for i in range(len(clip_start)):
            clip_dict = video.get_clip(clip_start[i], clip_end[i])
            if clip_dict is None or clip_dict["video"] is None:
              self.dataset._loaded_clip = None
              break
            loaded_clip_list.append(clip_dict)

          if self.dataset._loaded_clip is not None:
            for key in loaded_clip_list[0].keys():
              self.dataset._loaded_clip[key] = [x[key] for x in loaded_clip_list]

      else:  # single clip case
        if aug_index == 0:
          self.dataset._loaded_clip = video.get_clip(clip_start, clip_end)

      self.dataset._last_clip_end_time = clip_end

      video_is_null = (
        self.dataset._loaded_clip is None or self.dataset._loaded_clip["video"] is None
      )
      if (
        is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
      ) or video_is_null:
        self.dataset._loaded_video_label[0].close()
        self.dataset._loaded_video_label = None
        self.dataset._last_clip_end_time = None
        self.dataset._clip_sampler.reset()
        if video_is_null:
          print(f'Failed to load clip {video.name}; trial {i_try}')
          continue

      frames = self.dataset._loaded_clip["video"]
      audio_samples = self.dataset._loaded_clip["audio"]
      sample_dict = {
        "video": frames,
        "video_name": video.name,
        "video_index": video_index,
        "clip_index": clip_index,
        "aug_index": aug_index,
        **info_dict,
        **({"audio": audio_samples} if audio_samples is not None else {}),
      }

      if self.dataset._transform is not None:
        sample_dict = self.dataset._transform(sample_dict)

        if sample_dict is None:
          continue

      return sample_dict
    else:
      raise RuntimeError(f'Failed to load video after {self.dataset._MAX_CONSECUTIVE_FAILURES} retries.')

