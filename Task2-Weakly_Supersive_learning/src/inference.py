import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys

# 获取项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
task_dir = os.path.dirname(current_dir)

# 添加Task2-Weakly_Supersive_learning目录到Python搜索路径
sys.path.append(os.path.dirname(current_dir))

# 导入模块
from models import AffinityNet, propagate_labels
from utils import PetDataset, get_transforms
from utils import apply_crf, visualize_cam, visualize_affinity

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AffinityNet推理')
    
    # 数据集参数
    parser.add_argument('--data_root', type=str, default='./data/oxford-iiit-pet',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./output/inference',
                        help='输出目录')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='骨干网络: resnet50, resnet101')
    parser.add_argument('--num_classes', type=int, default=37,
                        help='类别数量')
    
    # 推理参数
    parser.add_argument('--split', type=str, default='test',
                        help='数据集分割: trainval, test')
    parser.add_argument('--cam_thresh', type=float, default=0.2,
                        help='CAM阈值')
    parser.add_argument('--num_iters', type=int, default=10,
                        help='标签传播迭代次数')
    parser.add_argument('--crf', action='store_true',
                        help='是否应用CRF后处理')
    
    # 其他参数
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批大小')
    
    return parser.parse_args()

def inference(args):
    """
    使用训练好的模型进行推理
    
    Args:
        args: 命令行参数
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = AffinityNet(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False
    )
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # 创建数据集
    transform = get_transforms('test')
    dataset = PetDataset(
        root=args.data_root,
        split=args.split,
        transform=transform
    )
    
    # 保存类别名称的字典
    class_names = dataset.classes
    
    # 进行推理
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='推理'):
            # 获取数据
            data = dataset[idx]
            image = data['image'].unsqueeze(0).to(device)
            image_id = data['image_id']
            
            # 前向传播
            outputs = model(image)
            
            # 获取CAM和亲和力
            cam = outputs['cam']  # [1, C, H, W]
            affinity = torch.sigmoid(outputs['affinity_map'])  # [1, 8, H, W]
            
            # 标签传播
            refined_cam = propagate_labels(
                affinity, cam, num_iter=args.num_iters, num_classes=args.num_classes
            )
            
            # 获取原始图像（用于可视化）
            orig_image = cv2.imread(os.path.join(dataset.images_dir, f'{image_id}.jpg'))
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            h, w, _ = orig_image.shape
            
            # 调整CAM大小
            cam_resized = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=True)
            refined_cam_resized = F.interpolate(refined_cam, size=(h, w), mode='bilinear', align_corners=True)
            
            # CRF后处理（如果需要）
            if args.crf:
                pred = apply_crf(orig_image, refined_cam_resized[0], num_classes=args.num_classes)
            else:
                pred = refined_cam_resized[0].argmax(0).cpu().numpy()
            
            # 保存结果
            # 1. 保存原始图像
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(orig_image)
            plt.title('原始图像')
            plt.axis('off')
            
            # 2. 可视化CAM
            cam_vis = visualize_cam(image[0], cam[0], threshold=args.cam_thresh)
            plt.subplot(2, 2, 2)
            plt.imshow(cam_vis)
            plt.title('CAM')
            plt.axis('off')
            
            # 3. 可视化传播后的CAM
            refined_cam_vis = visualize_cam(image[0], refined_cam[0], threshold=args.cam_thresh)
            plt.subplot(2, 2, 3)
            plt.imshow(refined_cam_vis)
            plt.title('传播后的CAM')
            plt.axis('off')
            
            # 4. 可视化分割结果
            plt.subplot(2, 2, 4)
            plt.imshow(pred, cmap='tab20')
            plt.title('分割结果')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'{image_id}_result.png'))
            plt.close()
            
            # 5. 可视化亲和力方向
            plt.figure(figsize=(15, 10))
            for d in range(8):
                plt.subplot(2, 4, d+1)
                aff_vis = visualize_affinity(image[0], affinity[0], direction=d)
                plt.imshow(aff_vis)
                plt.title(f'亲和力 方向 {d}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'{image_id}_affinity.png'))
            plt.close()
            
            # 6. 保存分割结果
            result_map = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(args.num_classes):
                mask = (pred == c)
                if mask.any():
                    # 使用类别名称作为分割结果的标签
                    class_name = class_names[c]
                    # 为每个类别使用不同的颜色
                    color = np.array([hash(class_name) % 256, 
                                     (hash(class_name) // 256) % 256, 
                                     (hash(class_name) // 65536) % 256], dtype=np.uint8)
                    result_map[mask] = color
                    
            # 叠加原始图像
            alpha = 0.5
            overlay = cv2.addWeighted(orig_image, 1 - alpha, result_map, alpha, 0)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(overlay)
            plt.title('分割结果叠加')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'{image_id}_overlay.png'))
            plt.close()

def main():
    """主函数"""
    args = parse_args()
    inference(args)

if __name__ == '__main__':
    main() 