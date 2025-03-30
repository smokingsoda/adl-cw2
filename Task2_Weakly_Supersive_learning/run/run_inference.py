import os
import sys
import argparse
import multiprocessing
import importlib.util

# 添加项目根目录到Python模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='AffinityNet Inference')
parser.add_argument('--data_root', type=str, required=True, help='Dataset root directory')
parser.add_argument('--model_path', type=str, required=True, help='Model weights path')
parser.add_argument('--output_dir', type=str, default='./output/inference', help='Output directory')
parser.add_argument('--split', type=str, default='test', help='Dataset split')
parser.add_argument('--num_images', type=int, default=10, help='Number of images to process')
parser.add_argument('--num_iters', type=int, default=10, help='Number of iterations for label propagation')
parser.add_argument('--crf', action='store_true', help='Use CRF post-processing')
parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='Backbone network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_classes', type=int, default=37, help='Number of classes')
parser.add_argument('--cam_thresh', type=float, default=0.2, help='CAM threshold for visualization')

if __name__ == '__main__':
    # 添加多进程支持
    multiprocessing.freeze_support()
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用importlib动态导入inference.py模块
    inference_path = os.path.join(project_root, 'src', 'inference.py')
    spec = importlib.util.spec_from_file_location("inference_module", inference_path)
    inference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference_module)
    
    # 执行inference.py中的main函数
    inference_module.main(args) 