import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

class RandomResizedCrop(object):
    """
    随机裁剪并调整大小
    """
    
    def __init__(self, size, scale=(0.5, 1.0), ratio=(3/4, 4/3)):
        """
        初始化
        
        Args:
            size: 输出大小
            scale: 面积比例范围
            ratio: 宽高比范围
        """
        self.size = size
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, img):
        """
        应用变换
        
        Args:
            img: 输入图像
        
        Returns:
            变换后的图像
        """
        width, height = img.size
        area = height * width
        
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if random.random() < 0.5:
                w, h = h, w
            
            if w <= width and h <= height:
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)
                
                img = img.crop((x1, y1, x1 + w, y1 + h))
                return img.resize((self.size, self.size), Image.BILINEAR)
        
        # 退化为中心裁剪
        scale = min(width, height) / max(width, height)
        if width < height:
            h = int(height * scale)
            y1 = (height - h) // 2
            img = img.crop((0, y1, width, y1 + h))
        else:
            w = int(width * scale)
            x1 = (width - w) // 2
            img = img.crop((x1, 0, x1 + w, height))
        
        return img.resize((self.size, self.size), Image.BILINEAR)

class RandomHorizontalFlip(object):
    """
    随机水平翻转
    """
    
    def __init__(self, p=0.5):
        """
        初始化
        
        Args:
            p: 翻转概率
        """
        self.p = p
    
    def __call__(self, img):
        """
        应用变换
        
        Args:
            img: 输入图像
        
        Returns:
            变换后的图像
        """
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

def get_transforms(split):
    """
    获取数据变换
    
    Args:
        split: 数据集分割
    
    Returns:
        变换
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        return transforms.Compose([
            RandomResizedCrop(448),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize
        ]) 