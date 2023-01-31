class Repeat:
  def __init__(self, num_repeats):
    self.num_repeats = num_repeats

  def __call__(self, image):
    return [image]+[image.copy() for _ in range(self.num_repeats-1)]


class ApplyTransformOnList:
  def __init__(self, transform):
    self.transform = transform

  def __call__(self, images):
    return [self.transform(image) for image in images]


class VideoToImage:
  def __init__(self):
    pass

  def __call__(self, video):
    return video[:, 0]
