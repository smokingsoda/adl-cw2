import os
import sys
import argparse
import multiprocessing
import importlib.util

# 添加项目根目录到Python模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Train AffinityNet')
parser.add_argument('--data_root', type=str, required=True, help='Dataset root directory')
parser.add_argument('--cam_dir', type=str, default='./data/cams', help='CAM files directory')
parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')

# 训练参数
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
parser.add_argument('--stage', type=int, default=1, help='Training stage')

# 模型参数
parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone network')
parser.add_argument('--num_classes', type=int, default=37, help='Number of classes in the dataset')
parser.add_argument('--lambda_aff', type=float, default=0.1, help='Affinity loss weight')

# CUDA优化参数
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练')
parser.add_argument('--cudnn_benchmark', action='store_true', help='启用cudnn.benchmark加速')
parser.add_argument('--empty_cache', action='store_true', help='每个epoch后清空CUDA缓存')
parser.add_argument('--cuda_prefetch', action='store_true', help='预热CUDA缓存以提高性能')

# 其他参数
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--resume', action='store_true', help='Resume training')
parser.add_argument('--model_path', type=str, default=None, help='Pre-trained model path')

if __name__ == '__main__':
    # 添加多进程支持
    multiprocessing.freeze_support()
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用importlib动态导入train.py模块
    train_path = os.path.join(project_root, 'src', 'train.py')
    spec = importlib.util.spec_from_file_location("train_module", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    # 执行train.py中的main函数
    train_module.main(args) 