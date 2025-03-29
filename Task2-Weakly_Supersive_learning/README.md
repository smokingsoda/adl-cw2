# 像素级语义亲和力学习复现 (Learning Pixel-level Semantic Affinity)

本项目受论文《Learning Pixel-level Semantic Affinity with Image-level Supervision》的启发，使用Oxford-IIIT宠物数据集实现弱监督语义分割。

## 论文简介

Ahn和Kwak在CVPR 2018发表的《Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation》论文提出了一种新的弱监督语义分割方法。该方法的主要创新点在于：

1. **语义亲和力学习**：提出了一种像素级语义亲和力网络(AffinityNet)，能够预测图像中任意两个位置之间的语义相似度。

2. **双阶段训练策略**：
   - 第一阶段：利用图像级标签训练分类网络，生成类激活图(CAM)
   - 第二阶段：利用CAM生成的像素级伪标签，训练AffinityNet学习像素间的语义亲和力

3. **随机游走传播**：利用学习到的亲和力矩阵，通过随机游走算法传播CAM的种子区域，显著提高分割精度

4. **仅使用图像级标签**：整个方法只需要图像级别的类别标签，无需像素级注释，大大降低了标注成本

本实现专注于复现论文的核心思想，使用Oxford-IIIT宠物数据集替代论文中使用的PASCAL VOC数据集。

论文链接：[https://arxiv.org/abs/1803.10464](https://arxiv.org/abs/1803.10464)

## 项目结构

```
Task2-Weakly_Supersive_learning/
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

## 训练脚本参数说明

`train.py`脚本实现了论文中的双阶段训练策略，以下是各参数的详细说明及其与论文的对应关系：

### 数据集参数

| 参数 | 类型 | 默认值 | 描述 | 论文代码对应 |
|------|------|--------|------|----------|
| `--data_root` | str | './data/oxford-iiit-pet' | 数据集根目录 | 论文代码实现使用 PASCAL VOC (--voc12_root)，非宠物数据集 |
| `--cam_dir` | str | './data/cams' | CAM文件目录 | - |
| `--output_dir` | str | './output' | 模型和日志输出目录 | - (实现细节) |

### 训练参数

| 参数 | 类型 | 默认值 | 描述 | 论文代码对应 |
|------|------|--------|------|----------|
| `--batch_size` | int | 8 | 批处理大小 | 论文代码实现中亲和力阶段 (train_aff.py) 使用 8。(论文未明确指定) |
| `--lr` | float | 0.001 | 学习率 | 论文代码实现中亲和力阶段基础学习率为 0.1，并使用 PolyOptimizer 进行复杂调度(不同层组 1x, 2x, 10x, 20x)。在本实现中第二阶段(亲和力阶段)，骨干网络使用的学习率是 args.lr * 0.1 |
| `--weight_decay` | float | 0.0005 | 权重衰减(L2正则化) | 论文代码实现中亲和力阶段(train_aff.py)使用 0.0005 (--wt_dec)，与本实现一致 |
| `--epochs` | int | 10 | 训练轮次 | 论文代码实现中亲和力阶段(train_aff.py)训练 8 个 epoch (--max_epoches)。本实现默认为 10 个 epoch |
| `--stage` | int | 1 | 训练阶段(1-分类,2-亲和力) | 对应论文的多阶段流程 (CAM生成 -> AffinityNet训练 -> 分割网络训练) |

### 模型参数

| 参数 | 类型 | 默认值 | 描述 | 论文代码对应 |
|------|------|--------|------|----------|
| `--backbone` | str | 'resnet50' | 骨干网络 | 论文使用 ResNet38。代码实现支持 ResNet38 (network.resnet38_aff) 和 VGG16 (network.vgg16_aff) 作为亲和力网络主干。分类网络代码也基于 ResNet38 |
| `--num_classes` | int | 37 | 类别数量 | - |
| `--lambda_aff` | float | 0.1 | 亲和力损失权重 | 在 AffinityLoss 类中实现，当 lambda_aff > 0 且提供了 affinity_mask 时计算亲和力损失。论文原代码实现中亲和力损失各部分(bg, fg, neg)的平衡通过 loss = bg_loss/4 + fg_loss/4 + neg_loss/2 硬编码实现 |

### 其他参数

| 参数 | 类型 | 默认值 | 描述 | 论文对应 |
|------|------|--------|------|----------|
| `--gpu_id` | int | 0 | 使用的GPU ID | - |
| `--num_workers` | int | 4 | 数据加载线程数 | - |
| `--seed` | int | 42 | 随机种子 | - |
| `--resume` | bool | False | 是否恢复训练 | - |

### 训练阶段与论文的对应关系

1. **第一阶段 (`--stage 1`)**：
   - 对应论文的"Classification Network Training"
   - 使用图像级别标签训练分类网络
   - 生成CAM(类激活图)作为下一阶段的伪标签

2. **第二阶段 (`--stage 2`)**：
   - 对应论文的"AffinityNet Training"
   - 利用第一阶段生成的CAM构建像素对的伪标签
   - 训练AffinityNet学习像素间的语义亲和力

## 安装

使用conda和提供的environment.yaml文件创建所需的环境：

```bash
conda env create -f environment.yaml
conda activate pixel-aff
```

或者，也可以使用pip安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集准备

1. 下载Oxford-IIIT宠物数据集：

数据集可以从[Academic Torrents](https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc1)下载。此数据集包含以下文件：
- images.tar.gz (791.92MB)
- annotations.tar.gz (19.17MB)

数据集简介：
> 我们创建了一个包含37个类别的宠物数据集，每个类别约有200张图像。图像在尺寸、姿势和光照方面有很大变化。所有图像都有相关的品种真值标注、头部ROI和像素级trimap分割。

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
python run_train.py --data_root data/oxford-iiit-pet --stage 1 --batch_size 8 --lr 0.001 --epochs 50
```

2. 第二阶段：训练亲和力模型

```bash
python run_train.py --data_root data/oxford-iiit-pet --stage 2 --cam_dir data/cams --batch_size 8 --lr 0.001 --epochs 50
```

### 推理

```bash
python run_inference.py --data_root data/oxford-iiit-pet --model_path output/stage2_resnet50/model_best.pth.tar --output_dir output/inference --crf
```

### 评估

```bash
python run_eval.py --data_root data/oxford-iiit-pet --model_path output/stage2_resnet50/model_best.pth.tar --output_dir output/eval --crf
```

## 实现细节
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

Oxford-IIIT Pet Dataset:
```
Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
``` 