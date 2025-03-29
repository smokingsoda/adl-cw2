import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet import ResNetBackbone

class AffinityNet(nn.Module):
    """
    AffinityNet模型，用于学习像素级语义亲和力
    
    参考论文: "Learning Pixel-level Semantic Affinity with Image-level Supervision"
    """
    
    def __init__(self, backbone='resnet50', num_classes=37, pretrained=True, dilated=True):
        super(AffinityNet, self).__init__()
        
        # 骨干网络
        self.backbone = ResNetBackbone(backbone=backbone, pretrained=pretrained, dilated=dilated)
        self.num_classes = num_classes
        
        # 定义用于计算亲和力的侧向连接
        self.lateral_conv1 = nn.Conv2d(self.backbone.channels['layer4'], 512, kernel_size=1, bias=False)
        self.lateral_conv2 = nn.Conv2d(self.backbone.channels['layer3'], 512, kernel_size=1, bias=False)
        self.lateral_conv3 = nn.Conv2d(self.backbone.channels['layer2'], 512, kernel_size=1, bias=False)
        
        # 亲和力映射头
        self.affinity_head = nn.Sequential(
            nn.Conv2d(512 * 3, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 8, kernel_size=1)  # 8个方向的亲和力
        )
        
        # CAM分类器头
        self.cam_head = nn.Conv2d(self.backbone.channels['layer4'], num_classes, kernel_size=1)
        
    def forward(self, x):
        """前向传播"""
        # 获取特征图
        features = self.backbone(x)
        
        # 获取多尺度特征
        x4 = features['layer4']  # 1/32
        x3 = features['layer3']  # 1/16
        x2 = features['layer2']  # 1/8
        
        # 侧向连接
        p4 = self.lateral_conv1(x4)
        p3 = self.lateral_conv2(x3)
        p2 = self.lateral_conv3(x2)
        
        # 上采样
        p4_up = F.interpolate(p4, size=p2.size()[2:], mode='bilinear', align_corners=True)
        p3_up = F.interpolate(p3, size=p2.size()[2:], mode='bilinear', align_corners=True)
        
        # 多尺度特征融合
        fused = torch.cat([p4_up, p3_up, p2], dim=1)
        
        # 计算亲和力图
        affinity_map = self.affinity_head(fused)  # [B, 8, H, W]
        
        # 计算CAM（用于分类监督）
        cam = self.cam_head(x4)  # [B, C, H, W]
        cam_logits = F.adaptive_avg_pool2d(cam, 1).squeeze(-1).squeeze(-1)  # [B, C]
        cam = F.interpolate(cam, size=p2.size()[2:], mode='bilinear', align_corners=True)
        
        return {
            'affinity_map': affinity_map,
            'cam': cam,
            'cam_logits': cam_logits
        }
    
    def get_parameter_groups(self):
        """获取参数组，用于设置不同的学习率"""
        groups = (
            list(self.backbone.parameters()),
            list(self.lateral_conv1.parameters()) + 
            list(self.lateral_conv2.parameters()) + 
            list(self.lateral_conv3.parameters()) +
            list(self.affinity_head.parameters()) +
            list(self.cam_head.parameters())
        )
        return groups

class AffinityLoss(nn.Module):
    """
    亲和力损失函数
    
    包括分类损失和亲和力损失
    """
    
    def __init__(self, num_classes=37, lambda_aff=0.1):
        super(AffinityLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_aff = lambda_aff
        self.classification_loss = nn.MultiLabelSoftMarginLoss()
        
    def forward(self, outputs, targets):
        """
        计算损失
        
        Args:
            outputs: 模型输出字典，包括'affinity_map', 'cam', 'cam_logits'
            targets: 目标字典，包括'label'（图像级标签）, 'affinity_mask'（亲和力掩码）
        
        Returns:
            总损失
        """
        # 分类损失
        cls_loss = self.classification_loss(outputs['cam_logits'], targets['label'])
        
        # 亲和力损失
        if 'affinity_mask' in targets and self.lambda_aff > 0:
            # 获取亲和力图和掩码
            affinity_map = outputs['affinity_map']  # [B, 8, H, W]
            affinity_mask = targets['affinity_mask']  # [B, 8, H, W]
            
            # 计算亲和力损失
            valid_mask = (affinity_mask >= 0).float()  # 有效区域掩码
            diff = torch.abs(affinity_map - affinity_mask)
            aff_loss = (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            
            # 总损失
            loss = cls_loss + self.lambda_aff * aff_loss
            
            return {
                'loss': loss,
                'cls_loss': cls_loss,
                'aff_loss': aff_loss
            }
        else:
            # 仅分类损失
            return {
                'loss': cls_loss,
                'cls_loss': cls_loss,
                'aff_loss': torch.tensor(0.0, device=cls_loss.device)
            }

def propagate_labels(affinity, cam, num_iter=10, num_classes=37, alpha=0.5):
    """
    使用学习的亲和力传播标签
    
    Args:
        affinity: 亲和力图，形状为 [B, 8, H, W]
        cam: CAM图，形状为 [B, C, H, W]
        num_iter: 传播迭代次数
        num_classes: 类别数
        alpha: 传播权重
    
    Returns:
        传播后的CAM图
    """
    device = affinity.device
    batch_size, _, h, w = cam.shape
    
    # 设置像素邻居的偏移量（8个方向）
    offsets = torch.tensor([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0, 1],
        [1, -1],  [1, 0],  [1, 1]
    ], device=device)
    
    # 创建传播后的CAM图
    prop_cam = cam.clone()
    
    # 迭代传播
    for _ in range(num_iter):
        old_cam = prop_cam.clone()
        
        # 对每个方向执行传播
        for d in range(8):
            offset_y, offset_x = offsets[d]
            
            # 创建偏移网格
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device)
            )
            
            # 计算邻居坐标
            nbr_y = grid_y + offset_y
            nbr_x = grid_x + offset_x
            
            # 检查边界
            valid = (nbr_y >= 0) & (nbr_y < h) & (nbr_x >= 0) & (nbr_x < w)
            
            # 遍历每个样本和每个类别
            for b in range(batch_size):
                for c in range(num_classes):
                    # 获取当前方向的亲和力
                    curr_aff = affinity[b, d]
                    
                    # 考虑有效位置的亲和力传播
                    curr_aff = curr_aff * valid.float()
                    
                    # 对邻居执行索引
                    nbr_cam = old_cam[b, c][nbr_y[valid], nbr_x[valid]]
                    
                    # 基于亲和力更新当前位置
                    update = torch.zeros_like(prop_cam[b, c])
                    update[grid_y[valid], grid_x[valid]] = curr_aff[valid] * nbr_cam
                    
                    # 将更新应用于传播的CAM
                    prop_cam[b, c] = (1 - alpha) * prop_cam[b, c] + alpha * update
    
    return prop_cam 