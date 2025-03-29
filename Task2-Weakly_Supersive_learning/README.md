# 像素级语义亲和力学习复现 (Learning Pixel-level Semantic Affinity)

本项目是对论文《Learning Pixel-level Semantic Affinity with Image-level Supervision》的代码复现，使用Oxford-IIIT宠物数据集实现弱监督语义分割。

## 项目结构

```
Task2-Weakly_Supersive_learning/
│
├── data/                  # 数据集目录
│   └── oxford-iiit-pet/   # Oxford-IIIT宠物数据集
│
├── models/                # 模型定义
│   ├── affinitynet.py     # AffinityNet模型
│   └── resnet.py          # 修改的ResNet骨干网络
│
├── src/                   # 源代码
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   └── eval.py            # 评估脚本
│
├── utils/                 # 工具函数
│   ├── datasets.py        # 数据集加载
│   ├── transforms.py      # 数据增强
│   └── misc.py            # 其他辅助函数
│
├── requirements.txt       # 依赖项
└── README.md              # 项目说明
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据集准备

1. 下载Oxford-IIIT宠物数据集：

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

2. 解压数据集：

```bash
mkdir -p data/oxford-iiit-pet
tar -xzf images.tar.gz -C data/oxford-iiit-pet
tar -xzf annotations.tar.gz -C data/oxford-iiit-pet
```

## 使用方法

### 训练

训练分为两个阶段：

1. 第一阶段：训练分类器生成CAM

```bash
python src/train.py --data_root data/oxford-iiit-pet --stage 1 --batch_size 8 --lr 0.001 --epochs 50
```

2. 第二阶段：训练亲和力模型

```bash
python src/train.py --data_root data/oxford-iiit-pet --stage 2 --cam_dir data/cams --batch_size 8 --lr 0.001 --epochs 50
```

### 推理

```bash
python src/inference.py --data_root data/oxford-iiit-pet --model_path output/stage2_resnet50/model_best.pth.tar --output_dir output/inference --crf
```

### 评估

```bash
python src/eval.py --data_root data/oxford-iiit-pet --model_path output/stage2_resnet50/model_best.pth.tar --output_dir output/eval --crf
```

## 实现细节

本项目对原论文进行了复现，主要特点：

1. 使用Oxford-IIIT宠物数据集（37个类别）进行训练和评估
2. 实现了一个双阶段训练过程：
   - 第一阶段：使用图像级别的标签训练分类器，生成CAM
   - 第二阶段：利用CAM生成亲和力标签，训练亲和力模型
3. 支持不同的骨干网络（ResNet50、ResNet101）
4. 实现了标签传播算法，利用学习的亲和力改进分割
5. 支持CRF后处理以进一步优化分割边界

## 引用

```
Ahn, J., & Kwak, S. (2018). Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4981-4990).
``` 