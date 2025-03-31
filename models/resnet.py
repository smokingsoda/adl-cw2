import torchvision.models as models
from torch import nn

ResNet18 = models.resnet18(weights='DEFAULT')
ResNet18.fc = nn.Linear(512, 37)

