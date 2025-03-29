import os
import sys
import argparse
import multiprocessing
import importlib.util

# 添加项目根目录到Python模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Affinity Net训练')
parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
parser.add_argument('--cam_dir', type=str, default='./data/cams', help='CAM文件目录')
parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='训练阶段 (1-分类,2-亲和力)')
parser.add_argument('--batch_size', type=int, default=8, help='批大小')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='骨干网络')
parser.add_argument('--num_classes', type=int, default=37, help='类别数量')
parser.add_argument('--lambda_aff', type=float, default=0.1, help='亲和力损失权重')
parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--resume', action='store_true', help='是否恢复训练')
parser.add_argument('--model_path', type=str, default='', help='用于第二阶段的预训练模型路径')

if __name__ == '__main__':
    # 添加多进程支持
    multiprocessing.freeze_support()
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 动态导入train.py模块
    train_path = os.path.join(project_root, 'src', 'train.py')
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    # 执行train.py中的main函数
    train_module.main(args) 