import clip
from opacus.validators import ModuleValidator


net, _ = clip.load('ViT-B/16')
print(net.visual.conv1.__class__)
print(ModuleValidator.is_valid(net.visual.conv1))
