from pytorchvideo.models.hub.vision_transformers import mvit_base_16x4


def get_mvit(num_classes, pretrained):
  model = mvit_base_16x4(pretrained)
  return model
