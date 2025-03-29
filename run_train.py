import os
import sys
import argparse
import multiprocessing

# 添加项目根目录到Python模块搜索路径
project_root = os.path.dirname(os.path.abspath(__file__))
task_dir = os.path.join(project_root, "Task2-Weakly_Supersive_learning")
sys.path.append(task_dir)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='训练AffinityNet')
parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
parser.add_argument('--stage', type=int, default=1, help='训练阶段 (1 或 2)')
parser.add_argument('--batch_size', type=int, default=8, help='批大小')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--epochs', type=int, default=50, help='训练轮次')
parser.add_argument('--cam_dir', type=str, default='./data/cams', help='CAM目录（第二阶段）')
parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
parser.add_argument('--resume', action='store_true', help='恢复训练')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')

if __name__ == '__main__':
    # 添加多进程支持
    multiprocessing.freeze_support()
    
    # 解析命令行参数并导入train.py中的main函数
    args = parser.parse_args()
    
    # 导入并执行train.py中的main函数
    from src.train import main
    main() 