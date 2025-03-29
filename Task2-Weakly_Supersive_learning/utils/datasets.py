import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from .transforms import get_transforms
import json

class PetDataset(Dataset):
    """
    Oxford-IIIT宠物数据集
    """
    
    def __init__(self, root, split='trainval', transform=None, target_transform=None):
        """
        初始化牛津宠物数据集
        
        Args:
            root: 数据集根目录
            split: 数据集分割 ('trainval', 'test')
            transform: 输入图像变换
            target_transform: 标签变换
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 宠物品种类别（37个类别）
        self.classes = [
            'Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
            'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair',
            'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter',
            'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
            'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher',
            'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue',
            'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
            'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier',
            'yorkshire_terrier'
        ]
        self.num_classes = len(self.classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        
        # 图像和标签路径
        self.images_dir = os.path.join(root, 'images')
        self.annotations_dir = os.path.join(root, 'annotations')
        
        # 加载图像ID列表
        self.ids = self._load_image_ids()
        
        # 图像预处理（如果没有提供）
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_image_ids(self):
        """加载图像ID列表"""
        split_file = os.path.join(self.annotations_dir, f'{self.split}.txt')
        if not os.path.exists(split_file):
            # 如果分割文件不存在，创建一个临时分割
            return self._create_split()
            
        with open(split_file, 'r') as f:
            ids = [line.strip().split(' ')[0] for line in f.readlines()]
        return ids
    
    def _create_split(self):
        """创建训练/测试分割"""
        # 获取所有图像文件
        image_files = [f.split('.')[0] for f in os.listdir(self.images_dir) 
                       if f.endswith('.jpg')]
        
        # 按品种分组
        breed_to_images = {}
        for img_id in image_files:
            # 提取品种名称
            parts = img_id.split('_')
            if len(parts) >= 2:
                breed = "_".join(parts[:-1])
                if breed in self.class_to_idx:
                    if breed not in breed_to_images:
                        breed_to_images[breed] = []
                    breed_to_images[breed].append(img_id)
            
        # 为每个品种分配80%训练，20%测试
        train_ids = []
        test_ids = []
        
        for breed, images in breed_to_images.items():
            n_train = int(len(images) * 0.8)
            train_ids.extend(images[:n_train])
            test_ids.extend(images[n_train:])
        
        # 根据分割返回对应的ID
        if self.split == 'trainval':
            return train_ids
        else:
            return test_ids
    
    def _load_image(self, image_id):
        """加载图像"""
        image_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        return image
    
    def _load_labels(self, image_id):
        """
        加载图像级标签（单标签分类）
        
        从图像ID中提取品种信息
        """
        labels = np.zeros(self.num_classes, dtype=np.float32)
        
        # 从图像ID提取品种
        parts = image_id.split('_')
        if len(parts) >= 2:
            breed = "_".join(parts[:-1])
            if breed in self.class_to_idx:
                idx = self.class_to_idx[breed]
                labels[idx] = 1
        
        return labels
    
    def __getitem__(self, index):
        """获取数据样本"""
        image_id = self.ids[index]
        
        # 加载图像和标签
        image = self._load_image(image_id)
        labels = self._load_labels(image_id)
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        
        # 转换为张量
        labels = torch.from_numpy(labels)
        
        return {
            'image': image,
            'label': labels,
            'image_id': image_id
        }
    
    def __len__(self):
        """获取数据集大小"""
        return len(self.ids)

class AffinityPetDataset(Dataset):
    """
    宠物亲和力数据集，用于训练AffinityNet
    
    从CAM种子生成亲和力标签
    """
    
    def __init__(self, root, cam_dir, split='trainval', transform=None, threshold=0.3):
        """
        初始化亲和力数据集
        
        Args:
            root: 数据集根目录
            cam_dir: CAM文件目录
            split: 数据集分割
            transform: 输入图像变换
            threshold: CAM阈值
        """
        self.root = root
        self.cam_dir = cam_dir
        self.split = split
        self.transform = transform
        self.threshold = threshold
        
        # 宠物品种类别
        self.classes = [
            'Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
            'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair',
            'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter',
            'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin',
            'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher',
            'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue',
            'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
            'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier',
            'yorkshire_terrier'
        ]
        self.num_classes = len(self.classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        
        # 图像和标签路径
        self.images_dir = os.path.join(root, 'images')
        self.annotations_dir = os.path.join(root, 'annotations')
        
        # 加载图像ID列表
        self.ids = self._load_image_ids()
        
        # 图像预处理（如果没有提供）
        if self.transform is None:
            self.transform = get_transforms(split)
    
    def _load_image_ids(self):
        """加载图像ID列表"""
        split_file = os.path.join(self.annotations_dir, f'{self.split}.txt')
        if not os.path.exists(split_file):
            # 如果分割文件不存在，创建一个临时分割
            return self._create_split()
            
        with open(split_file, 'r') as f:
            ids = [line.strip().split(' ')[0] for line in f.readlines()]
        return ids
    
    def _create_split(self):
        """创建训练/测试分割"""
        # 获取所有图像文件
        image_files = [f.split('.')[0] for f in os.listdir(self.images_dir) 
                       if f.endswith('.jpg')]
        
        # 按品种分组
        breed_to_images = {}
        for img_id in image_files:
            # 提取品种名称
            parts = img_id.split('_')
            if len(parts) >= 2:
                breed = "_".join(parts[:-1])
                if breed in self.class_to_idx:
                    if breed not in breed_to_images:
                        breed_to_images[breed] = []
                    breed_to_images[breed].append(img_id)
            
        # 为每个品种分配80%训练，20%测试
        train_ids = []
        test_ids = []
        
        for breed, images in breed_to_images.items():
            n_train = int(len(images) * 0.8)
            train_ids.extend(images[:n_train])
            test_ids.extend(images[n_train:])
        
        # 根据分割返回对应的ID
        if self.split == 'trainval':
            return train_ids
        else:
            return test_ids
    
    def _load_image(self, image_id):
        """加载图像"""
        image_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        return image
    
    def _load_labels(self, image_id):
        """加载图像级标签"""
        labels = np.zeros(self.num_classes, dtype=np.float32)
        
        # 从图像ID提取品种
        parts = image_id.split('_')
        if len(parts) >= 2:
            breed = "_".join(parts[:-1])
            if breed in self.class_to_idx:
                idx = self.class_to_idx[breed]
                labels[idx] = 1
        
        return labels
    
    def _load_cam(self, image_id):
        """加载CAM"""
        cam_path = os.path.join(self.cam_dir, f'{image_id}.npy')
        cam = np.load(cam_path)  # [C, H, W]
        return cam
    
    def _generate_affinity_mask(self, image_id, image):
        """
        生成亲和力掩码 - 固定尺寸为56x56，匹配模型输出
        
        Args:
            image_id: 图像ID
            image: 图像数据
        
        Returns:
            亲和力掩码 [8, 56, 56]
        """
        # 固定输出分辨率为56x56
        target_size = (56, 56)
        
        # 加载CAM
        cam = self._load_cam(image_id)  # [C, H, W]
        labels = self._load_labels(image_id)  # [C]
        
        # 将CAM直接调整为目标尺寸(56x56)
        cam_resized = np.zeros((self.num_classes, *target_size), dtype=np.float32)
        
        # 只处理存在的类别
        for c in range(self.num_classes):
            if labels[c] > 0 and c < cam.shape[0]:  # 确保CAM中有这个类别
                # 调整CAM大小为固定尺寸
                cls_cam = Image.fromarray(cam[c])
                cls_cam = cls_cam.resize(target_size, Image.BILINEAR)
                cam_resized[c] = np.array(cls_cam)
        
        # 阈值处理CAM
        cam_thresholded = (cam_resized > self.threshold).astype(np.float32)
        
        # 生成亲和力掩码（8个方向）
        affinity_mask = np.zeros((8, *target_size), dtype=np.float32)
        
        # 8个方向的偏移量
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # 目标尺寸的高度和宽度
        h, w = target_size
        
        # 生成亲和力标签
        for d, (dy, dx) in enumerate(offsets):
            for y in range(h):
                for x in range(w):
                    # 邻居坐标
                    ny, nx = y + dy, x + dx
                    
                    # 检查边界
                    if 0 <= ny < h and 0 <= nx < w:
                        # 如果两个像素属于相同的类别，则亲和力为1
                        for c in range(self.num_classes):
                            if cam_thresholded[c, y, x] > 0 and cam_thresholded[c, ny, nx] > 0:
                                affinity_mask[d, y, x] = 1
                                break
                        
                        # 如果两个像素属于不同的类别，则亲和力为0
                        for c1 in range(self.num_classes):
                            if cam_thresholded[c1, y, x] > 0:
                                for c2 in range(self.num_classes):
                                    if c1 != c2 and cam_thresholded[c2, ny, nx] > 0:
                                        affinity_mask[d, y, x] = 0
                                        break
                    else:
                        # 边界外的像素亲和力标记为-1（忽略）
                        affinity_mask[d, y, x] = -1
        
        return affinity_mask
    
    def __getitem__(self, index):
        """获取数据样本"""
        image_id = self.ids[index]
        
        # 加载图像和标签
        image = self._load_image(image_id)
        labels = self._load_labels(image_id)
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        # 生成亲和力掩码
        affinity_mask = self._generate_affinity_mask(image_id, image)
        
        # 转换为张量
        labels = torch.from_numpy(labels)
        affinity_mask = torch.from_numpy(affinity_mask)
        
        return {
            'image': image,
            'label': labels,
            'affinity_mask': affinity_mask,
            'image_id': image_id
        }
    
    def __len__(self):
        """获取数据集大小"""
        return len(self.ids)

def get_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 工作线程数
    
    Returns:
        数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 