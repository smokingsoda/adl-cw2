# AffinityNet 模型包
from .affinitynet import AffinityNet, AffinityLoss, propagate_labels
from .resnet import ResNetBackbone

__all__ = ['AffinityNet', 'AffinityLoss', 'propagate_labels', 'ResNetBackbone'] 