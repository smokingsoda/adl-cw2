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
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from torch.utils.data import DataLoader
import torchvision.transforms as T

# 获取项目根目录路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 导入项目模块
from utils.datasets import PetDataset
from utils.transforms import get_transforms
from models.resnet import ResNetBackbone
from models.affinitynet import AffinityNet, propagate_labels

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Model evaluation')
    
    # 数据集相关参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split (train, val, test)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
                        
    # 模型相关参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone architecture (default: resnet50)')
    parser.add_argument('--num_classes', type=int, default=37,
                        help='Number of classes (default: 37)')
                        
    # 评估相关参数
    parser.add_argument('--num_images', type=int, default=0,
                        help='Number of images to evaluate (default: 0 = all images)')
    parser.add_argument('--num_iters', type=int, default=10,
                        help='Number of iterations for affinity propagation (default: 10)')
    parser.add_argument('--crf', action='store_true',
                        help='Apply CRF post-processing')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
                        
    # 硬件相关参数
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    
    return parser.parse_args()

def apply_crf(img, probs, num_classes=2):
    """
    应用条件随机场后处理
    
    参数:
        img: 原始RGB图像，形状为(H, W, 3)，值范围为[0, 255]
        probs: 概率图，形状为(C, H, W)，C为类别数量
        num_classes: 类别数量，默认为2（前景和背景）
    
    返回:
        mask: 优化后的分割掩码，形状为(H, W)，值范围为[0, num_classes-1]
    """
    h, w = img.shape[:2]
    
    # 将概率转换为合适的格式
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    
    # 确保概率的形状为(C, H, W)
    if probs.shape[0] != num_classes:
        raise ValueError(f"probability tensor channels ({probs.shape[0]}) do not match the number of classes ({num_classes})")
    
    # 打印输入概率的统计信息
    for c in range(num_classes):
        print(f"  Class {c}: min={probs[c].min():.4f}, max={probs[c].max():.4f}, mean={probs[c].mean():.4f}")
    
    # 创建CRF模型
    d = dcrf.DenseCRF2D(w, h, num_classes)
    
    # 设置一元势能（unary potentials）
    U = unary_from_softmax(probs)  # 从softmax概率创建一元势
    d.setUnaryEnergy(U)
    
    # 设置成对势能（pairwise potentials）
    # 这将创建与位置和颜色相关的势能
    # 位置相关势：保证空间上相邻的像素趋向于有相同的标签
    # 颜色相关势：保证颜色相似的像素趋向于有相同的标签
    d.addPairwiseGaussian(sxy=5, compat=3)  # 位置相关势（短距离）
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=img, compat=10)  # 颜色相关势
    
    # 执行推理
    Q = d.inference(10)  # 增加到10次迭代以获得更稳定的结果
    
    # 将结果转换为分割掩码
    MAP = np.argmax(Q, axis=0).reshape((h, w))
    
    return MAP

def propagate_labels(affinity_map, cam, num_iter=10, num_classes=37):
    """
    使用亲和力图传播标签
    
    参数:
        affinity_map: 亲和力图, 形状 [B, 8, H, W]
        cam: 类激活图, 形状 [B, C, H, W]
        num_iter: 传播迭代次数
        num_classes: 类别数量
        
    返回:
        refined_cam: 标签传播后的CAM, 形状 [B, C, H, W]
    """
    B, C, H, W = cam.shape
    
    # 添加背景通道
    bg_channel = (1.0 - torch.max(cam, dim=1, keepdim=True)[0]).clamp(min=0.0, max=1.0)
    refined_cam = torch.cat([bg_channel, cam], dim=1)  # [B, C+1, H, W]
    
    for i in range(num_iter):
        # 执行标签传播迭代
        refined_cam = refine_cam(refined_cam, affinity_map)
    
    # 移除背景通道
    return refined_cam[:, 1:, :, :]

def refine_cam(cam, affinity_map):
    """
    执行一次标签传播迭代
    
    参数:
        cam: 类激活图, 形状 [B, C+1, H, W]
        affinity_map: 亲和力图, 形状 [B, 8, H, W]
        
    返回:
        refined_cam: 传播后的CAM, 形状 [B, C+1, H, W]
    """
    B, C, H, W = cam.shape
    
    # 获取8个方向上的CAM值
    shifted_cams = []
    for i in range(8):
        shifted_cam = shift_cam(cam, i)
        shifted_cams.append(shifted_cam)
    
    # 加权求和
    refined_cam = cam.clone()
    
    # 扩展亲和力图维度以匹配cam
    aff_weight = affinity_map.unsqueeze(2)  # [B, 8, 1, H, W]
    
    # 对每个方向应用亲和力权重并传播
    for i in range(8):
        shifted_cam = shifted_cams[i].unsqueeze(1)  # [B, 1, C, H, W]
        weighted_cam = aff_weight[:, i:i+1] * shifted_cam  # [B, 1, C, H, W]
        refined_cam += weighted_cam.squeeze(1)  # [B, C, H, W]
    
    # 归一化
    refined_cam = F.normalize(refined_cam, p=1, dim=1)
    
    return refined_cam

def shift_cam(cam, direction_idx):
    """
    根据指定方向移动CAM
    
    参数:
        cam: 类激活图, 形状 [B, C, H, W]
        direction_idx: 方向索引，0-7表示8个方向
        
    返回:
        shifted_cam: 移动后的CAM, 形状 [B, C, H, W]
    """
    B, C, H, W = cam.shape
    
    # 8个方向: 右、右下、下、左下、左、左上、上、右上
    # 对应的移动: 
    # 0: (0, 1)  1: (1, 1)  2: (1, 0)  3: (1, -1)
    # 4: (0, -1) 5: (-1, -1) 6: (-1, 0) 7: (-1, 1)
    shifts = [(0, 1), (1, 1), (1, 0), (1, -1),
              (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    dy, dx = shifts[direction_idx]
    
    # 创建新的tensor
    shifted_cam = torch.zeros_like(cam)
    
    # 移动CAM
    if dx == 1:
        shifted_cam[:, :, :, 0:-1] = cam[:, :, :, 1:]
    elif dx == -1:
        shifted_cam[:, :, :, 1:] = cam[:, :, :, 0:-1]
    else:
        shifted_cam[:, :, :, :] = cam[:, :, :, :]
    
    if dy == 1:
        shifted_cam[:, :, 0:-1, :] = shifted_cam[:, :, 1:, :]
        shifted_cam[:, :, -1, :] = 0
    elif dy == -1:
        shifted_cam[:, :, 1:, :] = shifted_cam[:, :, 0:-1, :]
        shifted_cam[:, :, 0, :] = 0
    
    return shifted_cam

def refine_cams_with_bkg_v2(cam, batch_size=1):
    """
    添加背景通道，并使用SOFTMAX进行归一化
    
    参数:
        cam: 类激活图, 形状 [B, C, H, W]
        batch_size: 批次大小
        
    返回:
        refined_cam: 添加背景通道后的CAM, 形状 [B, C+1, H, W]
    """
    reshaped_cam = F.relu(cam)  # 首先ReLU去除负值
    
    # 获取每个位置的最大值
    max_val, _ = torch.max(reshaped_cam, dim=1, keepdim=True)
    
    # 创建背景通道: 1 - 最大类别激活
    bkg = F.relu(1 - max_val)
    
    # 连接背景通道
    refined_cam = torch.cat([bkg, reshaped_cam], dim=1)
    
    # 使用softmax归一化
    refined_cam = F.softmax(refined_cam * 10, dim=1)  # 乘以10使得分布更加极端
    
    return refined_cam

def evaluate(args):
    """
    评估模型性能
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 加载模型
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = AffinityNet(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False
    )
    model.to(device)
    
    if os.path.isfile(args.model_path):
        print(f"=> Loading model: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"=> No checkpoint found at: {args.model_path}")
        return
    
    # 切换到评估模式
    model.eval()
    
    # 加载数据集
    transform = get_transforms('test')
    dataset = PetDataset(
        root=args.data_root,
        split=args.split,
        transform=transform
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset size: {len(dataset)} images")
    
    # 初始化指标计算
    all_precision = []
    all_recall = []
    all_f1 = []
    all_iou = []
    
    # 处理每个图像
    for i, data in enumerate(dataloader):
        if args.num_images > 0 and i >= args.num_images:
            break
            
        images = data['image'].to(device)
        labels = data['label']
        image_id = data['image_id'][0]
        label = labels[0].cpu().numpy()
        
        print(f"\nProcessing image {i+1}/{min(args.num_images, len(dataset)) if args.num_images > 0 else len(dataset)}: {image_id}")
        
        # 生成CAM
        with torch.no_grad():
            outputs = model(images)
            cam = outputs['cam']
            affinity = torch.sigmoid(outputs['affinity_map'])
            refined_cam = propagate_labels(
                affinity, cam, num_iter=args.num_iters, num_classes=args.num_classes
            )
        
        # 加载ground truth掩码
        gt_mask_path = os.path.join(args.data_root, 'annotations', 'trimaps', f'{image_id}.png')
        if not os.path.exists(gt_mask_path):
            print(f"Warning: GT mask not found for {image_id}, skipping.")
            continue
            
        gt_mask = np.array(Image.open(gt_mask_path))
        # 将trimap转换为二进制掩码，其中2=边界，1=前景，3=背景
        # 对于二值分割，我们将1视为前景，其余视为背景
        gt_binary = np.zeros_like(gt_mask)
        gt_binary[gt_mask == 1] = 1  # 只有1（前景）被视为正类
        
        # 调整CAM大小以匹配真值掩码
        h, w = gt_mask.shape
        refined_cam_resized = F.interpolate(refined_cam, size=(h, w), mode='bilinear', align_corners=True)
        
        # 获取原始图像（用于CRF）
        orig_image = cv2.imread(os.path.join(dataset.images_dir, f'{image_id}.jpg'))
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        orig_image = cv2.resize(orig_image, (w, h))
        
        # 获取活跃类别索引
        class_indices = np.where(label > 0)[0]
        print(f"Image: {image_id}, Active classes: {class_indices}, Label: {label}")
        
        # 如果没有找到激活的类别，使用预测概率最高的类别
        if len(class_indices) == 0:
            cam_logits = outputs['cam_logits'][0].cpu().numpy()
            top_class = np.argmax(cam_logits)
            class_indices = [top_class]
            print(f"No active classes found in label, using top predicted class: {top_class}")
        
        # ===== 方法1：使用argmax策略生成掩码 (推荐) =====
        # 获取每个像素预测的类别
        # 背景标签为-1，前景类别从0开始
        pred_classes = torch.argmax(refined_cam_resized[0], dim=0)
        
        # 创建二值掩码
        pred_mask_argmax = torch.zeros_like(pred_classes, dtype=torch.int64)
        
        # 对每个活跃类别，将对应类别的像素标记为前景
        for c in class_indices:
            pred_mask_argmax[pred_classes == c] = 1
        
        # 转换为numpy
        pred_mask_argmax = pred_mask_argmax.cpu().numpy()
        
        # 计算每个类别的CAM值
        print("CAM statistics for each class:")
        for c in range(min(5, refined_cam_resized.shape[1])):  # 只显示前5个类别
            class_cam = refined_cam_resized[0, c]
            print(f"  Class {c}: min={class_cam.min().item():.4f}, max={class_cam.max().item():.4f}, "
                  f"mean={class_cam.mean().item():.4f}, pixels={torch.sum(pred_classes == c).item()}")
        
        # 获取对应类别的CAM
        foreground_cam = torch.zeros_like(refined_cam_resized[0, 0])
        for c in class_indices:
            foreground_cam = torch.maximum(foreground_cam, refined_cam_resized[0, c])
        
        # 归一化CAM到[0,1]范围（用于可视化）
        min_val = foreground_cam.min()
        max_val = foreground_cam.max()
        print(f"CAM value range: [{min_val.item():.4f}, {max_val.item():.4f}]")
        
        # 标准化后Sigmoid激活（用于可视化）
        foreground_cam = torch.sigmoid(foreground_cam)
        print(f"CAM value range after sigmoid: [{foreground_cam.min().item():.4f}, {foreground_cam.max().item():.4f}]")
        
        # 默认使用argmax方法
        pred_mask = pred_mask_argmax
        argmax_pixels = np.sum(pred_mask_argmax)
        print(f"Argmax method: {argmax_pixels} foreground pixels ({argmax_pixels/pred_mask_argmax.size*100:.2f}%)")
        
        # 如果argmax方法没有检测到任何前景，或者检测到过多前景（>95%），警告用户
        if argmax_pixels == 0 or argmax_pixels > pred_mask_argmax.size * 0.95:
            print("Warning: Argmax method may be unreliable (no foreground or too much foreground)")
        
        # CRF后处理（如果需要）
        if args.crf:
            try:
                # 创建多类别预测概率
                # 首先创建背景概率
                bg_prob = 1.0 - foreground_cam.unsqueeze(0)
                
                # 然后创建前景类别概率 
                # 对每个活跃类别，将其CAM作为该类别的概率
                class_probs = []
                for c in class_indices:
                    class_prob = refined_cam_resized[0, c].unsqueeze(0)
                    if class_prob.max() > 0:
                        class_prob = class_prob / class_prob.max()  # 归一化
                    class_probs.append(class_prob)
                
                # 如果没有活跃类别，创建一个伪类别
                if not class_probs:
                    class_probs = [foreground_cam.unsqueeze(0)]
                
                # 合并所有概率 [bg, class1, class2, ...]
                all_probs = torch.cat([bg_prob] + class_probs, dim=0)
                
                # 确保概率和为1
                all_probs = F.softmax(all_probs * 10, dim=0)  # 加权以增强对比度
                
                # 打印CRF输入信息
                print(f"CRF input shape: {all_probs.shape}, range: [{all_probs.min().item():.4f}, {all_probs.max().item():.4f}]")
                
                # 确保背景和前景都有合理的概率分布
                if all_probs[0].min().item() > 0.95 or all_probs[0].max().item() < 0.05:
                    print("WARNING: Background probability is too extreme. Adjusting...")
                    # 调整背景概率以确保有合理的分布
                    bg_weight = 0.7
                    fg_weight = 1.0 - bg_weight
                    all_probs[0] = all_probs[0] * bg_weight
                    # 相应调整前景概率
                    for i in range(1, all_probs.shape[0]):
                        all_probs[i] = all_probs[i] * fg_weight
                    # 重新归一化
                    all_probs = F.normalize(all_probs, p=1, dim=0)
                
                # 应用CRF
                crf_mask = apply_crf(orig_image, all_probs, num_classes=len(class_indices)+1)
                
                # 生成二值掩码：任何非背景类别都视为前景
                crf_binary_mask = (crf_mask > 0).astype(np.int64)
                print(f"CRF mask unique values: {np.unique(crf_mask)}, foreground pixels: {np.sum(crf_binary_mask)} ({np.sum(crf_binary_mask)/crf_binary_mask.size*100:.2f}%)")
                
                # 如果CRF结果不合理（全黑或全白），回退到原始掩码
                if np.sum(crf_binary_mask) == 0 or np.sum(crf_binary_mask) > crf_binary_mask.size * 0.95:
                    print("CRF result unreasonable, falling back to original mask")
                else:
                    pred_mask = crf_binary_mask
                    print("Using CRF result as final prediction")
            except Exception as e:
                print(f"CRF error: {e}, falling back to original mask")
        
        # 打印最终预测掩码统计信息
        print(f"Final predicted mask: foreground pixels: {np.sum(pred_mask)}, total pixels: {pred_mask.size}, percentage: {np.sum(pred_mask)/pred_mask.size*100:.2f}%")
        
        # 计算指标
        precision = precision_score(gt_binary.flatten(), pred_mask.flatten(), average='binary', zero_division=0)
        recall = recall_score(gt_binary.flatten(), pred_mask.flatten(), average='binary', zero_division=0)
        f1 = f1_score(gt_binary.flatten(), pred_mask.flatten(), average='binary', zero_division=0)
        iou = jaccard_score(gt_binary.flatten(), pred_mask.flatten(), average='binary', zero_division=0)
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_iou.append(iou)
        
        print(f"Image {image_id}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, IoU={iou:.4f}")
        
        # 可视化
        # 创建一个可视化图像，显示原图、真值、初始CAM和最终预测
        plt.figure(figsize=(20, 10))
        
        # 读取原始图像用于可视化
        original_img = cv2.imread(os.path.join(dataset.images_dir, f'{image_id}.jpg'))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (w, h))
        
        # 1. 原始图像
        plt.subplot(2, 3, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # 2. 真值掩码
        plt.subplot(2, 3, 2)
        plt.imshow(gt_binary, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        # 3. 原始CAM
        plt.subplot(2, 3, 3)
        plt.imshow(foreground_cam.cpu().numpy(), cmap='jet')
        plt.title('CAM (Sigmoid)')
        plt.axis('off')
        
        # 4. ArgMax方法
        plt.subplot(2, 3, 4)
        plt.imshow(pred_mask_argmax, cmap='gray')
        plt.title(f'ArgMax Method ({argmax_pixels/pred_mask_argmax.size*100:.2f}%)')
        plt.axis('off')
        
        # 5. 最终预测 (CRF或原始)
        plt.subplot(2, 3, 5)
        final_method = "ArgMax"
        if args.crf and not np.array_equal(pred_mask, pred_mask_argmax):
            final_method = "CRF"
            
        plt.imshow(pred_mask, cmap='gray')
        plt.title(f'Final Prediction ({final_method})\n{np.sum(pred_mask)/pred_mask.size*100:.2f}% foreground')
        plt.axis('off')
        
        # 6. 叠加显示
        plt.subplot(2, 3, 6)
        overlay = original_img.copy()
        overlay[pred_mask == 1] = [0, 255, 0]  # 使用绿色标记预测的前景
        
        # 混合显示
        alpha = 0.7  # 增加透明度让前景更清晰
        combined = cv2.addWeighted(original_img, 1-alpha, overlay, alpha, 0)
        
        plt.imshow(combined)
        plt.title(f'Overlay ({final_method})\nP={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, IoU={iou:.4f}')
        plt.axis('off')
        
        # 保存可视化结果
        plt.tight_layout()
        vis_path = os.path.join(vis_dir, f'{image_id}_result.png')
        plt.savefig(vis_path)
        plt.close()
        
        # 也保存单独的预测掩码
        mask_path = os.path.join(args.output_dir, f'{image_id}_mask.png')
        cv2.imwrite(mask_path, pred_mask * 255)
    
    # 计算平均指标
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    avg_iou = np.mean(all_iou)
    
    print("\n==== Evaluation Results ====")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    
    # 保存结果到文件
    result_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(result_path, 'w') as f:
        f.write("==== Evaluation Results ====\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write("\n==== Detailed Metrics for Each Image ====\n")
        for i, image_id in enumerate(dataloader.dataset.ids[:min(args.num_images, len(dataloader.dataset))]):
            f.write(f"Image {image_id}: Precision={all_precision[i]:.4f}, Recall={all_recall[i]:.4f}, F1={all_f1[i]:.4f}, IoU={all_iou[i]:.4f}\n")

def main(args):
    """
    主函数
    """
    # 检查命令行参数
    if not hasattr(args, 'backbone'):
        args.backbone = 'resnet50'
    if not hasattr(args, 'num_classes'):
        args.num_classes = 37
    if not hasattr(args, 'batch_size'):
        args.batch_size = 1
    if not hasattr(args, 'num_workers'):
        args.num_workers = 4
        
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 评估模型
    evaluate(args)

if __name__ == '__main__':
    main() 