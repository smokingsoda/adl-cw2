# 弱监督学习源代码包
from .train import main as train_main
from .inference import main as inference_main
from .eval import main as eval_main

__all__ = [
    'train_main',  # 训练主函数
    'inference_main',  # 推理主函数
    'eval_main'  # 评估主函数
] 