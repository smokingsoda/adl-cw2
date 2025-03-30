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
        生成亲和力掩码 - 使用更高效的方法实现论文中的策略
        
        Args:
            image_id: 图像ID
            image: 图像数据
        
        Returns:
            亲和力掩码 [8, 56, 56]
        """
        import torch
        from skimage import img_as_ubyte
        
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
            HAS_CRF = True
        except ImportError:
            print("CRF未安装，使用简化方法生成亲和力掩码")
            HAS_CRF = False
        
        # 固定输出分辨率为56x56
        target_size = (56, 56)
        
        # 加载CAM
        cam = self._load_cam(image_id)  # [C, H, W]
        labels = self._load_labels(image_id)  # [C]
        
        # 将CAM调整为目标尺寸
        cam_resized = np.zeros((self.num_classes, *target_size), dtype=np.float32)
        
        # 只处理存在的类别
        active_classes = []
        for c in range(self.num_classes):
            if labels[c] > 0 and c < cam.shape[0]:  # 确保CAM中有这个类别
                active_classes.append(c)
                # 调整CAM大小为固定尺寸
                cls_cam = Image.fromarray(cam[c])
                cls_cam = cls_cam.resize(target_size, Image.BILINEAR)
                cam_resized[c] = np.array(cls_cam)
        
        # 如果没有活跃类别，返回全零掩码
        if len(active_classes) == 0:
            return np.zeros((8, *target_size), dtype=np.float32)
        
        # 转换为PyTorch张量，方便操作
        cam_torch = torch.from_numpy(cam_resized).float()
        
        # 对CAM应用softmax得到类别概率
        cam_exp = torch.exp(cam_torch)
        cam_prob = cam_exp / (cam_exp.sum(dim=0, keepdim=True) + 1e-8)
        
        # 创建背景通道 (均值为0.5的置信度)
        bg_channel = 0.5 * torch.ones((1, *target_size), dtype=torch.float32)
        
        # CRF处理（如果可用）生成置信区域
        if HAS_CRF:
            # 原始图像的小版本，用于CRF
            if isinstance(image, torch.Tensor):
                # 如果是张量，转换回numpy
                img_np = image.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
            else:
                # 如果已经是PIL图像
                img_np = np.array(image.resize(target_size, Image.BILINEAR))
            
            # 准备CRF输入
            cam_with_bg = torch.cat([bg_channel, cam_prob], dim=0)  # [C+1, H, W]
            
            # 创建低alpha和高alpha的CRF
            d_low = dcrf.DenseCRF2D(target_size[1], target_size[0], self.num_classes + 1)
            d_high = dcrf.DenseCRF2D(target_size[1], target_size[0], self.num_classes + 1)
            
            # 设置一元势能
            U = unary_from_softmax(cam_with_bg.numpy())
            d_low.setUnaryEnergy(U)
            d_high.setUnaryEnergy(U)
            
            # 设置二元势能
            # 低alpha (4)
            d_low.addPairwiseGaussian(sxy=3, compat=3)
            d_low.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img_np, compat=4)
            
            # 高alpha (32)
            d_high.addPairwiseGaussian(sxy=3, compat=3)
            d_high.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img_np, compat=32)
            
            # 推理
            Q_low = np.array(d_low.inference(10)).reshape(self.num_classes + 1, *target_size)
            Q_high = np.array(d_high.inference(10)).reshape(self.num_classes + 1, *target_size)
            
            # 转换回torch
            Q_low_torch = torch.from_numpy(Q_low).float()
            Q_high_torch = torch.from_numpy(Q_high).float()
            
            # 置信区域定义
            # 1. 低alpha CRF中某个类别的得分显著高于其他类别
            bg_conf_thresh = 0.3
            fg_conf_thresh = 0.5
            
            # 置信前景掩码 [C, H, W]
            conf_fg_mask = torch.zeros((self.num_classes, *target_size), dtype=torch.bool)
            # 置信背景掩码 [H, W]
            conf_bg_mask = (Q_high_torch[0] > bg_conf_thresh)  # 高alpha背景得分高
            
            # 对每个前景类别
            for i, c in enumerate(active_classes):
                # 取低alpha CRF中类别c+1的分数(+1因为有背景通道)
                cls_score = Q_low_torch[c+1]
                # 该类别与其他所有类别的最高分数比较
                other_max = torch.max(torch.cat([
                    Q_low_torch[:c+1], Q_low_torch[c+2:]
                ]), dim=0)[0]
                # 置信前景：该类别得分高于阈值，且显著高于其他所有类别
                conf_fg_mask[c] = (cls_score > fg_conf_thresh) & (cls_score > other_max + 0.3)
        else:
            # 如果没有CRF，使用简单阈值定义置信区域
            conf_fg_mask = cam_prob > 0.7  # 置信前景：概率大于0.7
            conf_bg_mask = cam_prob.max(dim=0)[0] < 0.3  # 置信背景：所有类别概率都小于0.3
            
        # 生成亲和力掩码
        affinity_mask = np.zeros((8, *target_size), dtype=np.float32)
        
        # 将掩码转换为numpy
        conf_fg_mask_np = conf_fg_mask.numpy()
        conf_bg_mask_np = conf_bg_mask.numpy()
        
        # 创建标记图：每个像素属于哪个类别（0=背景，1...C=前景类别）
        # -1表示中性区域（不够置信）
        label_map = np.full(target_size, -1, dtype=np.int32)
        
        # 背景区域标记为0
        label_map[conf_bg_mask_np] = 0
        
        # 前景区域标记为类别ID
        for c in active_classes:
            label_map[conf_fg_mask_np[c]] = c + 1  # +1因为0是背景
        
        # 8个方向的偏移量
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # 使用高效的数组操作生成亲和力标签
        h, w = target_size
        
        # 生成亲和力标签 - 使用矢量化操作
        for d, (dy, dx) in enumerate(offsets):
            # 创建移位后的标记图
            shifted_map = np.full(target_size, -1, dtype=np.int32)
            
            # 有效区域索引
            y_src_start = max(0, dy)
            y_src_end = h if dy <= 0 else h + dy
            x_src_start = max(0, dx)
            x_src_end = w if dx <= 0 else w + dx
            
            y_dst_start = max(0, -dy)
            y_dst_end = h if dy >= 0 else h - dy
            x_dst_start = max(0, -dx)
            x_dst_end = w if dx >= 0 else w - dx
            
            # 复制有效部分
            shifted_map[y_src_start:y_src_end, x_src_start:x_src_end] = \
                label_map[y_dst_start:y_dst_end, x_dst_start:x_dst_end]
            
            # 中性区域掩码（-1表示中性）
            neutral_mask = (label_map == -1) | (shifted_map == -1)
            
            # 在中性区域，亲和力标签设为-1（忽略）
            affinity_mask[d][neutral_mask] = -1
            
            # 非中性区域
            valid_mask = ~neutral_mask
            
            # 同一类别区域，亲和力为1
            same_class = (label_map == shifted_map) & valid_mask
            affinity_mask[d][same_class] = 1
            
            # 不同类别区域，亲和力为0
            diff_class = (label_map != shifted_map) & valid_mask
            affinity_mask[d][diff_class] = 0
        
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