import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def init_weights(m):
    """
    初始化模型权重
    
    Args:
        m: 模型模块
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    """
    保存检查点
    
    Args:
        state: 状态字典
        is_best: 是否是最佳模型
        save_dir: 保存目录
        filename: 文件名
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(save_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)

def load_checkpoint(model, optimizer, scheduler, filepath):
    """
    加载检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        filepath: 文件路径
    
    Returns:
        开始轮次和最佳性能
    """
    if not os.path.exists(filepath):
        return 0, float('inf')
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['epoch'], checkpoint['best_perf']

def visualize_cam(image, cam, threshold=0.3, alpha=0.5):
    """
    可视化CAM
    
    Args:
        image: 输入图像，形状为 [3, H, W]
        cam: CAM，形状为 [C, H, W]
        threshold: 阈值
        alpha: 混合透明度
    
    Returns:
        可视化图像
    """
    # 将图像转换为numpy，并调整形状
    image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
    
    # 获取图像尺寸
    img_h, img_w = image.shape[:2]
    
    # 创建热图
    cam_sum = cam.sum(0)  # [H, W]
    cam_sum = (cam_sum - cam_sum.min()) / (cam_sum.max() - cam_sum.min() + 1e-8)  # 归一化到 [0, 1]
    
    # 应用阈值
    cam_sum = (cam_sum > threshold).float()
    
    # 转换为热图
    cam_sum = cam_sum.cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_sum), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # 调整热图尺寸与原始图像匹配
    if heatmap.shape[:2] != (img_h, img_w):
        heatmap = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    
    # 混合
    mixed = (1 - alpha) * image + alpha * heatmap
    return np.uint8(255 * mixed)

def visualize_affinity(image, affinity, direction=0):
    """
    可视化亲和力
    
    Args:
        image: 输入图像，形状为 [3, H, W]
        affinity: 亲和力，形状为 [8, H, W]
        direction: 要可视化的方向（0-7）
    
    Returns:
        可视化图像
    """
    # 将图像转换为numpy，并调整形状
    image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
    
    # 获取图像尺寸
    img_h, img_w = image.shape[:2]
    
    # 获取指定方向的亲和力
    aff = affinity[direction].cpu().numpy()  # [H, W]
    aff = (aff - aff.min()) / (aff.max() - aff.min() + 1e-8)  # 归一化到 [0, 1]
    
    # 转换为热图
    aff_map = cv2.applyColorMap(np.uint8(255 * aff), cv2.COLORMAP_JET)  # [H, W, 3]
    aff_map = cv2.cvtColor(aff_map, cv2.COLOR_BGR2RGB) / 255.0
    
    # 调整热图尺寸与原始图像匹配
    if aff_map.shape[:2] != (img_h, img_w):
        aff_map = cv2.resize(aff_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    
    # 混合
    mixed = 0.7 * image + 0.3 * aff_map
    return np.uint8(255 * mixed)

def apply_crf(image, cam, num_classes=21, iter_max=10):
    """
    应用CRF后处理
    
    Args:
        image: 输入图像，形状为 [H, W, 3]，值范围为 [0, 255]
        cam: CAM，形状为 [C, H, W]，值范围为 [0, 1]
        num_classes: 类别数
        iter_max: 最大迭代次数
    
    Returns:
        后处理后的分割图，形状为 [H, W]
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("pydensecrf not installed, skip CRF.")
        return np.argmax(cam, axis=0)
    
    h, w = cam.shape[1:]
    cam = cam.cpu().numpy()
    
    # 确保CAM形状正确
    if cam.shape[0] != num_classes:
        temp = np.zeros((num_classes, h, w), dtype=np.float32)
        temp[:cam.shape[0]] = cam
        cam = temp
    
    # 应用softmax
    cam = cam.transpose(1, 2, 0)  # [H, W, C]
    cam = np.exp(cam) / np.sum(np.exp(cam), axis=2, keepdims=True)
    cam = cam.transpose(2, 0, 1)  # [C, H, W]
    
    # 创建CRF
    d = dcrf.DenseCRF2D(w, h, num_classes)
    
    # 设置一元势
    U = unary_from_softmax(cam)
    d.setUnaryEnergy(U)
    
    # 设置二元势
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    
    # 推理
    Q = d.inference(iter_max)
    Q = np.array(Q).reshape(num_classes, h, w)
    
    # 返回最可能的类别
    return np.argmax(Q, axis=0) 