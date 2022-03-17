import timm


def get_convnext(num_classes, pretrained):
  model = timm.create_model('convnext_base_in22ft1k', pretrained=pretrained, num_classes=num_classes)
  return model
