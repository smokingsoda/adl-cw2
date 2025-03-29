import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import sys

# 获取项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
task_dir = os.path.dirname(current_dir)

# 添加Task2-Weakly_Supersive_learning目录到Python搜索路径
sys.path.append(os.path.dirname(current_dir))

# 导入模块
from models import AffinityNet, propagate_labels
from utils import PetDataset, get_transforms, apply_crf

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AffinityNet评估')
    
    # 数据集参数
    parser.add_argument('--data_root', type=str, default='./data/oxford-iiit-pet',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./output/eval',
                        help='输出目录')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='骨干网络: resnet50, resnet101')
    parser.add_argument('--num_classes', type=int, default=37,
                        help='类别数量')
    
    # 评估参数
    parser.add_argument('--split', type=str, default='test',
                        help='数据集分割: trainval, test')
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

def evaluate(args):
    """
    评估模型性能
    
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
    
    # 初始化指标
    precisions = []
    recalls = []
    f1_scores = []
    ious = []
    
    # 进行评估
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='评估'):
            # 获取数据
            data = dataset[idx]
            image = data['image'].unsqueeze(0).to(device)
            label = data['label'].numpy()  # [C]
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
            
            # 获取原始图像和分割真值
            try:
                # 尝试加载分割真值
                mask_path = os.path.join(args.data_root, 'annotations', 'trimaps', f'{image_id}.png')
                if not os.path.exists(mask_path):
                    continue
                
                gt_mask = np.array(Image.open(mask_path))
                
                # Oxford-IIIT Pet数据集中，trimap标注：1=宠物，2=背景，3=边界
                # 我们将边界当作背景处理
                gt_mask = (gt_mask == 1).astype(np.int64)  # 1 = 宠物，0 = 背景或边界
                
                # 调整CAM大小以匹配真值掩码
                h, w = gt_mask.shape
                refined_cam_resized = F.interpolate(refined_cam, size=(h, w), mode='bilinear', align_corners=True)
                
                # 获取原始图像（用于CRF）
                orig_image = cv2.imread(os.path.join(dataset.images_dir, f'{image_id}.jpg'))
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                orig_image = cv2.resize(orig_image, (w, h))
                
                # 根据标签预测前景
                # 获取对应类别的CAM（根据图像级标签）
                class_indices = np.where(label > 0)[0]
                foreground_cam = torch.zeros_like(refined_cam_resized[0, 0])
                
                for c in class_indices:
                    foreground_cam = torch.maximum(foreground_cam, refined_cam_resized[0, c])
                
                # CRF后处理（如果需要）
                if args.crf:
                    # 创建二分类CAM：0=背景，1=前景
                    binary_cam = torch.stack([1 - foreground_cam, foreground_cam], dim=0).unsqueeze(0)
                    pred_mask = apply_crf(orig_image, binary_cam[0], num_classes=2)
                else:
                    # 阈值处理
                    pred_mask = (foreground_cam > 0.5).cpu().numpy().astype(np.int64)
                
                # 计算指标
                precision = precision_score(gt_mask.flatten(), pred_mask.flatten(), average='binary', zero_division=0)
                recall = recall_score(gt_mask.flatten(), pred_mask.flatten(), average='binary', zero_division=0)
                f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), average='binary', zero_division=0)
                iou = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), average='binary', zero_division=0)
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                ious.append(iou)
                
                # 保存可视化结果
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(orig_image)
                plt.title('原始图像')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask, cmap='gray')
                plt.title('真值掩码')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask, cmap='gray')
                plt.title(f'预测掩码 (IoU: {iou:.4f})')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'{image_id}_eval.png'))
                plt.close()
            
            except Exception as e:
                print(f"处理图像 {image_id} 时出错: {e}")
                continue
    
    # 计算平均指标
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    mean_iou = np.mean(ious)
    
    # 打印结果
    print(f"评估结果 (split={args.split}, crf={args.crf}):")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # 保存结果
    results = {
        'precision': mean_precision,
        'recall': mean_recall,
        'f1_score': mean_f1,
        'iou': mean_iou,
        'split': args.split,
        'crf': args.crf,
        'model_path': args.model_path
    }
    
    np.save(os.path.join(args.output_dir, 'results.npy'), results)
    
    # 生成图表
    plt.figure(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1 Score', 'IoU']
    values = [mean_precision, mean_recall, mean_f1, mean_iou]
    
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    plt.ylim(0, 1.0)
    plt.title('评估指标')
    plt.ylabel('分数')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics.png'))
    plt.close()

def main(args=None):
    """主函数
    
    Args:
        args: 命令行参数对象。如果为None，则使用parse_args解析
    """
    if args is None:
        args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main() 