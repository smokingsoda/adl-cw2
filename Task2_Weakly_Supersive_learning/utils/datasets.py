import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from .transforms import get_transforms
import json
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

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
    
    def __init__(self, root, cam_dir, split='trainval', transform=None, threshold=0.3, img_size=224):
        """
        初始化亲和力数据集
        
        Args:
            root: 数据集根目录
            cam_dir: CAM文件目录
            split: 数据集分割
            transform: 输入图像变换
            threshold: CAM阈值
            img_size: 图像大小
        """
        self.root = root
        self.cam_dir = cam_dir
        self.split = split
        self.transform = transform
        self.threshold = threshold
        self.img_size = img_size
        
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
        生成亲和力掩码
        使用条件随机场（CRF）生成更平滑的亲和力掩码
        """
        # 读取原始图像
        img_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        img = Image.open(img_path)
        img_np = np.asarray(img, dtype=np.uint8)
        
        # 确保数组是C连续的
        img_np = np.ascontiguousarray(img_np)
        
        # 读取掩码
        mask_path = os.path.join(self.annotations_dir, 'trimaps', f'{image_id}.png')
        mask = Image.open(mask_path)
        mask_np = np.asarray(mask, dtype=np.int32)
        mask_np = np.ascontiguousarray(mask_np)  # 确保掩码也是C连续的
        
        # 将trimap转换为二值掩码
        binary_mask = np.zeros_like(mask_np, dtype=np.int32)
        binary_mask[mask_np == 1] = 1  # 前景
        binary_mask[mask_np == 2] = 0  # 边界（视为背景）
        binary_mask[mask_np == 3] = 0  # 背景
        
        # 将图像和掩码的大小调整为模型的输入大小
        if image.shape[-2:] != (self.img_size, self.img_size):
            h, w = image.shape[-2:]
            img_np = cv2.resize(img_np, (w, h))
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 生成亲和力掩码
        h, w = binary_mask.shape
        d_low = dcrf.DenseCRF2D(w, h, 2)  # 2类：前景和背景
        
        # 将二值掩码转换为概率
        U_low = np.zeros((2, h * w), dtype=np.float32)  # 修改为2D数组
        U_low[0, binary_mask.ravel() == 0] = 0.9  # 背景
        U_low[0, binary_mask.ravel() == 1] = 0.1  # 前景
        U_low[1, binary_mask.ravel() == 0] = 0.1  # 背景
        U_low[1, binary_mask.ravel() == 1] = 0.9  # 前景
        
        # 保证概率分布有效
        U_low = np.ascontiguousarray(U_low)  # 确保概率数组是C连续的
        
        # 设置一元势
        d_low.setUnaryEnergy(-np.log(U_low))
        
        # 生成亲和力掩码
        affinity_mask = np.zeros((8, *binary_mask.shape), dtype=np.float32)
        
        # 8个方向的偏移量
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # 使用高效的数组操作生成亲和力标签
        for d, (dy, dx) in enumerate(offsets):
            # 创建移位后的标记图
            shifted_map = np.full(binary_mask.shape, -1, dtype=np.int32)
            
            # 计算源和目标的索引范围
            if dy >= 0:
                y_src_start, y_src_end = 0, h - dy
                y_dst_start, y_dst_end = dy, h
            else:
                y_src_start, y_src_end = -dy, h
                y_dst_start, y_dst_end = 0, h + dy
                
            if dx >= 0:
                x_src_start, x_src_end = 0, w - dx
                x_dst_start, x_dst_end = dx, w
            else:
                x_src_start, x_src_end = -dx, w
                x_dst_start, x_dst_end = 0, w + dx
            
            # 复制有效部分
            shifted_map[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
                binary_mask[y_src_start:y_src_end, x_src_start:x_src_end]
            
            # 中性区域掩码（-1表示中性）
            neutral_mask = (binary_mask == -1) | (shifted_map == -1)
            
            # 在中性区域，亲和力标签设为-1（忽略）
            affinity_mask[d][neutral_mask] = -1
            
            # 非中性区域
            valid_mask = ~neutral_mask
            
            # 同一类别区域，亲和力为1
            same_class = (binary_mask == shifted_map) & valid_mask
            affinity_mask[d][same_class] = 1
            
            # 不同类别区域，亲和力为0
            diff_class = (binary_mask != shifted_map) & valid_mask
            affinity_mask[d][diff_class] = 0
        
        # 将亲和力掩码调整为模型输出大小（1/8）
        h_out = h // 8
        w_out = w // 8
        affinity_mask_resized = np.zeros((8, h_out, w_out), dtype=np.float32)
        for d in range(8):
            affinity_mask_resized[d] = cv2.resize(affinity_mask[d], (w_out, h_out), interpolation=cv2.INTER_NEAREST)
        
        return affinity_mask_resized
    
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