import os
import sys
import argparse
import multiprocessing
import importlib.util

# 添加项目根目录到Python模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='AffinityNet评估')
parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
parser.add_argument('--output_dir', type=str, default='./output/eval', help='输出目录')
parser.add_argument('--split', type=str, default='test', help='数据集分割')
parser.add_argument('--num_iters', type=int, default=10, help='标签传播迭代次数')
parser.add_argument('--crf', action='store_true', help='使用CRF后处理')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_images', type=int, default=0, help='要评估的图像数量(0表示全部)')

if __name__ == '__main__':
    # 添加多进程支持
    multiprocessing.freeze_support()
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用importlib动态导入eval.py模块
    eval_path = os.path.join(project_root, 'src', 'eval.py')
    spec = importlib.util.spec_from_file_location("eval_module", eval_path)
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    
    # 执行eval.py中的main函数
    eval_module.main(args) 