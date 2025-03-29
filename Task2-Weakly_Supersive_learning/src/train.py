import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from models.affinitynet import AffinityNet, AffinityLoss
from utils.datasets import PetDataset, AffinityPetDataset, get_dataloader
from utils.transforms import get_transforms
from utils.misc import init_weights, save_checkpoint, load_checkpoint, visualize_cam, visualize_affinity

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练AffinityNet')
    
    # 数据集参数
    parser.add_argument('--data_root', type=str, default='./data/oxford-iiit-pet',
                        help='数据集根目录')
    parser.add_argument('--cam_dir', type=str, default='./data/cams',
                        help='CAM文件目录（用于第二阶段训练）')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮次')
    parser.add_argument('--stage', type=int, default=1,
                        help='训练阶段: 1-分类阶段, 2-亲和力阶段')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='骨干网络: resnet50, resnet101')
    parser.add_argument('--num_classes', type=int, default=37,
                        help='类别数量')
    parser.add_argument('--lambda_aff', type=float, default=0.1,
                        help='亲和力损失权重')
    
    # 其他参数
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--resume', action='store_true',
                        help='恢复训练')
    
    return parser.parse_args()

def train_stage1(args):
    """
    第一阶段训练: 分类阶段
    
    使用图像级别的标签训练分类器并生成CAM
    """
    # 创建输出目录
    save_dir = os.path.join(args.output_dir, f'stage1_{args.backbone}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = AffinityNet(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=True
    )
    
    # 初始化权重
    model.apply(init_weights)
    model = model.to(device)
    
    # 创建数据集和数据加载器
    train_transform = get_transforms('trainval')
    val_transform = get_transforms('test')
    
    train_dataset = PetDataset(
        root=args.data_root,
        split='trainval',
        transform=train_transform
    )
    
    val_dataset = PetDataset(
        root=args.data_root,
        split='test',
        transform=val_transform
    )
    
    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 40],
        gamma=0.1
    )
    
    # 创建损失函数
    criterion = AffinityLoss(num_classes=args.num_classes, lambda_aff=0.0)
    
    # 恢复训练（如果需要）
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth.tar')
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_loss, train_cls_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # 验证
        val_loss, val_cls_loss = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录损失
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/cls_loss', train_cls_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/cls_loss', val_cls_loss, epoch)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_perf': best_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best,
            save_dir
        )
    
    # 生成CAM
    print('生成CAM...')
    os.makedirs(args.cam_dir, exist_ok=True)
    generate_cams(model, train_loader, args.cam_dir, device)
    generate_cams(model, val_loader, args.cam_dir, device)
    
    writer.close()

def train_stage2(args):
    """
    第二阶段训练: 亲和力阶段
    
    使用CAM生成亲和力标签，训练亲和力模型
    """
    # 创建输出目录
    save_dir = os.path.join(args.output_dir, f'stage2_{args.backbone}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = AffinityNet(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=True
    )
    
    # 初始化权重
    model.apply(init_weights)
    
    # 加载第一阶段模型
    stage1_path = os.path.join(args.output_dir, f'stage1_{args.backbone}', 'model_best.pth.tar')
    if os.path.exists(stage1_path):
        checkpoint = torch.load(stage1_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'加载第一阶段模型: {stage1_path}')
    
    model = model.to(device)
    
    # 创建数据集和数据加载器
    train_transform = get_transforms('trainval')
    val_transform = get_transforms('test')
    
    train_dataset = AffinityPetDataset(
        root=args.data_root,
        cam_dir=args.cam_dir,
        split='trainval',
        transform=train_transform,
        threshold=0.3
    )
    
    val_dataset = AffinityPetDataset(
        root=args.data_root,
        cam_dir=args.cam_dir,
        split='test',
        transform=val_transform,
        threshold=0.3
    )
    
    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 获取骨干网络和其他参数组
    param_groups = model.get_parameter_groups()
    
    # 创建优化器，为骨干网络设置较小的学习率
    optimizer = optim.SGD(
        [
            {'params': param_groups[0], 'lr': args.lr * 0.1},
            {'params': param_groups[1], 'lr': args.lr}
        ],
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 40],
        gamma=0.1
    )
    
    # 创建损失函数
    criterion = AffinityLoss(num_classes=args.num_classes, lambda_aff=args.lambda_aff)
    
    # 恢复训练（如果需要）
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth.tar')
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_loss, train_cls_loss, train_aff_loss = train_epoch_aff(
            model, train_loader, optimizer, criterion, device
        )
        
        # 验证
        val_loss, val_cls_loss, val_aff_loss = validate_aff(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录损失
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/cls_loss', train_cls_loss, epoch)
        writer.add_scalar('train/aff_loss', train_aff_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/cls_loss', val_cls_loss, epoch)
        writer.add_scalar('val/aff_loss', val_aff_loss, epoch)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_perf': best_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best,
            save_dir
        )
    
    writer.close()

def train_epoch(model, data_loader, optimizer, criterion, device):
    """
    训练一个epoch（阶段1：分类）
    
    Args:
        model: 模型
        data_loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    
    for batch in tqdm(data_loader, desc='训练'):
        # 获取数据
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss_dict = criterion(outputs, {'label': labels})
        loss = loss_dict['loss']
        cls_loss = loss_dict['cls_loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    
    return avg_loss, avg_cls_loss

def train_epoch_aff(model, data_loader, optimizer, criterion, device):
    """
    训练一个epoch（阶段2：亲和力）
    
    Args:
        model: 模型
        data_loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_aff_loss = 0.0
    
    for batch in tqdm(data_loader, desc='训练'):
        # 获取数据
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        affinity_mask = batch['affinity_mask'].to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss_dict = criterion(outputs, {'label': labels, 'affinity_mask': affinity_mask})
        loss = loss_dict['loss']
        cls_loss = loss_dict['cls_loss']
        aff_loss = loss_dict['aff_loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_aff_loss += aff_loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    avg_aff_loss = total_aff_loss / len(data_loader)
    
    return avg_loss, avg_cls_loss, avg_aff_loss

def validate(model, data_loader, criterion, device):
    """
    验证（阶段1：分类）
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='验证'):
            # 获取数据
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss_dict = criterion(outputs, {'label': labels})
            loss = loss_dict['loss']
            cls_loss = loss_dict['cls_loss']
            
            # 累计损失
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    
    return avg_loss, avg_cls_loss

def validate_aff(model, data_loader, criterion, device):
    """
    验证（阶段2：亲和力）
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_aff_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='验证'):
            # 获取数据
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            affinity_mask = batch['affinity_mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss_dict = criterion(outputs, {'label': labels, 'affinity_mask': affinity_mask})
            loss = loss_dict['loss']
            cls_loss = loss_dict['cls_loss']
            aff_loss = loss_dict['aff_loss']
            
            # 累计损失
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_aff_loss += aff_loss.item()
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    avg_cls_loss = total_cls_loss / len(data_loader)
    avg_aff_loss = total_aff_loss / len(data_loader)
    
    return avg_loss, avg_cls_loss, avg_aff_loss

def generate_cams(model, data_loader, save_dir, device):
    """
    生成CAM
    
    Args:
        model: 模型
        data_loader: 数据加载器
        save_dir: 保存目录
        device: 设备
    """
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='生成CAM'):
            # 获取数据
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            
            # 前向传播
            outputs = model(images)
            
            # 获取CAM
            cams = outputs['cam'].cpu().numpy()  # [B, C, H, W]
            
            # 保存CAM
            for i, image_id in enumerate(image_ids):
                cam_path = os.path.join(save_dir, f'{image_id}.npy')
                np.save(cam_path, cams[i])

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 根据阶段选择训练函数
    if args.stage == 1:
        train_stage1(args)
    elif args.stage == 2:
        train_stage2(args)
    else:
        raise ValueError(f"无效的训练阶段: {args.stage}")

if __name__ == '__main__':
    main() 