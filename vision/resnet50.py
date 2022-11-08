import torch
import torchvision.models as models


resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
image = torch.randn(1, 3, 224, 224)
out = resnet50(image)
print(image.flatten()[0:10])
print(out[0][0:10])

images = torch.randn(3, 3, 224, 224)
images[0] = image
print(images.shape)
out = resnet50(images)
print(images.flatten()[0:10])
print(out[0][0:10])
