import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """修改后的ResNet骨干网络，用于提取特征"""
    
    def __init__(self, backbone='resnet50', pretrained=True, dilated=True):
        super(ResNetBackbone, self).__init__()
        
        # 选择预训练的ResNet模型
        if backbone == 'resnet50':
            orig_resnet = models.resnet50(pretrained=pretrained)
            self.output_stride = 8 if dilated else 32
        elif backbone == 'resnet101':
            orig_resnet = models.resnet101(pretrained=pretrained)
            self.output_stride = 8 if dilated else 32
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")

        # 使用ResNet的初始层
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        
        # 获取ResNet的各个阶段
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        
        # 修改layer3和layer4以增加膨胀率（如果需要）
        if dilated:
            self.layer3 = self._dilate_layer(orig_resnet.layer3, 2)
            self.layer4 = self._dilate_layer(orig_resnet.layer4, 4)
        else:
            self.layer3 = orig_resnet.layer3
            self.layer4 = orig_resnet.layer4
            
        # 特征维度
        self.channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }
    
    def _dilate_layer(self, layer, dilation):
        """增加ResNet层的膨胀率并移除下采样步长"""
        # 更彻底地修改第一个块的步长
        # 直接设置Bottleneck的stride属性
        layer[0].stride = (1, 1)
        
        # 修改第一个瓶颈块中的下采样步长
        if hasattr(layer[0], 'downsample') and layer[0].downsample is not None:
            # 获取下采样层
            downsample = layer[0].downsample
            # 修改下采样层中的卷积步长
            if isinstance(downsample[0], nn.Conv2d):
                downsample[0].stride = (1, 1)
        
        # 修改第一个瓶颈块内部卷积的步长
        if hasattr(layer[0], 'conv2'):
            layer[0].conv2.stride = (1, 1)
        if hasattr(layer[0], 'conv1') and layer[0].conv1.stride == (2, 2):
            layer[0].conv1.stride = (1, 1)
        
        # 打印检查确认步长已修改
        print(f"Modified layer stride: {layer[0].stride}")
        if hasattr(layer[0], 'downsample') and layer[0].downsample is not None:
            if isinstance(layer[0].downsample[0], nn.Conv2d):
                print(f"Modified downsample stride: {layer[0].downsample[0].stride}")
        
        # 修改所有块中3x3卷积的膨胀率和填充
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (dilation, dilation)
                    m.dilation = (dilation, dilation)
        return layer
    
    def forward(self, x):
        """前向传播，返回多尺度特征"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 提取不同尺度的特征
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        return {
            'layer1': layer1,
            'layer2': layer2,
            'layer3': layer3,
            'layer4': layer4
        } 