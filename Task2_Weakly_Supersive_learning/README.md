# 像素级语义亲和力学习 (Learning Pixel-level Semantic Affinity)

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
Task2_Weakly_Supersive_learning/
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
├── run/                   # 运行脚本
│   ├── run_train.py       # 训练启动脚本
│   ├── run_inference.py   # 推理启动脚本
│   └── run_eval.py        # 评估启动脚本  
│
└── README.md              # 项目说明
```

## 骨干网络修改

为了使ResNet更适合语义分割任务，我们对标准ResNet进行了以下修改：

### 修改后的Layer3

- **输入**: H/8 × W/8 × 512
- **结构**: 基于原始layer3，包含6个瓶颈残差块
- **修改**:
  - 移除了第一个瓶颈块中的下采样步长，使空间分辨率保持不变
  - 所有3×3卷积层的dilation(膨胀率)设置为2，padding调整为(2,2)
  - 这增大了卷积的感受野，同时保持特征图的空间尺寸
- **输出**: H/8 × W/8 × 1024 (保持空间分辨率不变)

### 修改后的Layer4

- **输入**: H/8 × W/8 × 1024
- **结构**: 基于原始layer4，包含3个瓶颈残差块
- **修改**:
  - 移除了第一个瓶颈块中的下采样步长，使空间分辨率保持不变
  - 所有3×3卷积层的dilation设置为4，padding调整为(4,4)
  - 这进一步增大了感受野，使模型能捕获更大范围的上下文信息
- **输出**: H/8 × W/8 × 2048 (保持空间分辨率不变)

### 修改的好处

1. **更高的特征图分辨率**: 输出步长从原来的32减小到8，保留了更多的空间细节
2. **更大的感受野**: 通过膨胀卷积，模型可以看到更广泛的上下文信息
3. **不增加参数量**: 仅修改了现有卷积层的属性，没有引入额外参数
4. **更适合分割任务**: 高分辨率特征图对于像素级预测任务至关重要

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
| `--batch_size` | int | 8 | 批处理大小 | 论文代码实现中亲和力阶段 (train_aff.py) 使用 8|
| `--lr` | float | 0.001 | 学习率 | 论文代码实现中亲和力阶段基础学习率为 0.1，并使用 PolyOptimizer 进行复杂调度(不同层组 1x, 2x, 10x, 20x)。在本实现中第二阶段(亲和力阶段)，骨干网络使用的学习率是 args.lr * 0.1 |
| `--weight_decay` | float | 0.0005 | 权重衰减(L2正则化) | 论文代码实现中亲和力阶段(train_aff.py)使用 0.0005 (--wt_dec)，与本实现一致 |
| `--epochs` | int | 10 | 训练轮次 | 论文代码实现中亲和力阶段(train_aff.py)训练 8 个 epoch (--max_epoches)。本实现默认为 10 个 epoch |
| `--stage` | int | 1 | 训练阶段(1-分类,2-亲和力) | 对应论文的多阶段流程 (CAM生成 -> AffinityNet训练 -> 分割网络训练) |

### 模型参数

| 参数 | 类型 | 默认值 | 描述 | 论文代码对应 |
|------|------|--------|------|----------|
| `--backbone` | str | 'resnet50' | 骨干网络 | 论文使用 ResNet38。本项目采用ResNet50 |
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

## 实验与使用方法

本节介绍了如何使用项目代码进行弱监督语义分割的训练、推理与评估实验。

### 准备工作

在开始实验之前，请完成以下准备工作：

```bash
# 创建并激活conda环境
conda env create -f environment.yaml
conda activate pixel-aff

# 创建必要的目录
mkdir -p data/oxford-iiit-pet data/cams output
```

### 数据集准备

下载并解压Oxford-IIIT宠物数据集：

```bash
# 下载数据集（如果尚未下载）
# 从 https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc1 下载

# 解压数据集
tar -xzf images.tar.gz -C data/oxford-iiit-pet
tar -xzf annotations.tar.gz -C data/oxford-iiit-pet
```

### CUDA优化选项

为获得更好的训练性能，特别是在显存有限的情况下，可以使用以下CUDA优化选项：

| 选项 | 作用 | 建议使用场景 |
|------|------|------------|
| `--use_amp` | 启用自动混合精度训练，使用FP16加速计算，减少显存占用 | 几乎所有情况下都建议使用 |
| `--cudnn_benchmark` | 自动寻找最佳卷积算法，加速训练 | 输入尺寸固定时使用 |
| `--empty_cache` | 每个epoch后清空CUDA缓存，减少内存碎片 | 显存不足时使用 |
| `--cuda_prefetch` | 预热CUDA缓存，优化内存分配 | 长时间训练时使用 |
| `--batch_size` | 调整批处理大小，根据GPU显存适当设置 | 8GB显存建议8-16，16GB显存建议16-32 |

### 阶段 1: 训练分类网络并生成 CAM

此阶段训练一个基于图像级标签的分类器，并生成后续阶段所需的类激活图 (CAM)。

```bash
# 创建输出目录
mkdir -p output/stage1_resnet50

# 运行第一阶段训练（使用CUDA优化）
python Task2_Weakly_Supersive_learning/run/run_train.py \
    --data_root ./data/oxford-iiit-pet \
    --stage 1 \
    --backbone resnet50 \
    --output_dir ./output \
    --cam_dir ./data/cams \
    --batch_size 16 \
    --lr 0.001 \
    --epochs 50 \
    --use_amp \
    --cudnn_benchmark \
    --empty_cache \
    --cuda_prefetch \
    --gpu_id 0
```

**预期输出:**
- 训练好的分类模型权重: `./output/stage1_resnet50/model_best.pth.tar`
- 类激活图(CAM): 存储在`./data/cams`目录下

### 阶段 2: 训练 AffinityNet

此阶段利用阶段1生成的CAM（处理为伪亲和力标签）来训练AffinityNet。

```bash
# 创建第二阶段输出目录
mkdir -p output/stage2_resnet50

# 运行第二阶段训练（使用CUDA优化）
python Task2_Weakly_Supersive_learning/run/run_train.py \
    --data_root ./data/oxford-iiit-pet \
    --stage 2 \
    --backbone resnet50 \
    --output_dir ./output \
    --cam_dir ./data/cams \
    --batch_size 16 \
    --lr 0.001 \
    --epochs 10 \
    --lambda_aff 0.1 \
    --model_path ./output/stage1_resnet50/model_best.pth.tar \
    --use_amp \
    --cudnn_benchmark \
    --empty_cache \
    --cuda_prefetch \
    --gpu_id 0
```

**预期输出:**
- 训练好的AffinityNet模型权重: `./output/stage2_resnet50/model_best.pth.tar`

### 阶段 3: 模型推理

使用训练好的亲和力模型进行推理：

```bash
# 创建推理输出目录
mkdir -p output/inference

# 运行推理
python Task2_Weakly_Supersive_learning/run/run_inference.py \
    --data_root ./data/oxford-iiit-pet \
    --model_path ./output/stage2_resnet50/model_best.pth.tar \
    --output_dir ./output/inference \
    --backbone resnet50 \
    --split test \
    --num_iters 10 \
    --crf \
    --gpu_id 0
```

### 阶段 4: 模型评估

评估模型性能：

```bash
# 创建评估输出目录
mkdir -p output/eval_with_crf output/eval_no_crf

# 运行评估（带CRF后处理）
python Task2_Weakly_Supersive_learning/run/run_eval.py \
    --data_root ./data/oxford-iiit-pet \
    --model_path ./output/stage2_resnet50/model_best.pth.tar \
    --output_dir ./output/eval_with_crf \
    --split test \
    --num_iters 10 \
    --crf \
    --gpu_id 0 \
    --num_images 20

# 运行评估（不带CRF后处理）
python Task2_Weakly_Supersive_learning/run/run_eval.py \
    --data_root ./data/oxford-iiit-pet \
    --model_path ./output/stage2_resnet50/model_best.pth.tar \
    --output_dir ./output/eval_no_crf \
    --split test \
    --num_iters 10 \
    --gpu_id 0 \
    --num_images 20
```

**预期输出:**
- 评估结果文件（`results.npy`）包含Mean Precision, Recall, F1, IoU等数值
- 评估指标可视化（`metrics.png`）
- 每张测试图像的分割结果可视化（`*_eval.png`）

### 阶段 5: 消融实验

为了深入理解框架各组件和超参数的影响，可以进行以下消融研究：

#### 5.1 亲和力损失权重实验

验证亲和力学习的重要性：

```bash
# 以不同的lambda_aff值运行训练
python Task2_Weakly_Supersive_learning/run/run_train.py \
    --data_root ./data/oxford-iiit-pet \
    --stage 2 \
    --backbone resnet50 \
    --output_dir ./output \
    --cam_dir ./data/cams \
    --batch_size 8 \
    --lr 0.001 \
    --epochs 10 \
    --lambda_aff 0 \    # 尝试不同值：0, 0.01, 0.1, 1.0
    --model_path ./output/stage1_resnet50/model_best.pth.tar \
    --gpu_id 0
```

#### 5.2 标签传播迭代次数实验

研究标签传播步骤的效果及迭代次数影响：

```bash
# 以不同的num_iters值运行评估
python Task2_Weakly_Supersive_learning/run/run_eval.py \
    --data_root ./data/oxford-iiit-pet \
    --model_path ./output/stage2_resnet50/model_best.pth.tar \
    --output_dir ./output/eval_iters_5 \
    --split test \
    --num_iters 5 \
    --crf \
    --gpu_id 0 \
    --num_images 20
```

#### 5.3 CRF后处理实验

量化CRF对分割边界精度的提升效果（已在阶段4中进行）。

#### 5.4 (可选) 骨干网络实验

探究更强大骨干网络的影响：

```bash
# 使用ResNet101进行训练
python Task2_Weakly_Supersive_learning/run/run_train.py \
    --data_root ./data/oxford-iiit-pet \
    --stage 1 \
    --backbone resnet101 \
    --output_dir ./output \
    --cam_dir ./data/cams \
    --batch_size 8 \
    --lr 0.001 \
    --epochs 50 \
    --gpu_id 0
```
macOS和Linux系统可以直接使用上述bash命令。

#### Windows (PowerShell)

**环境准备：**
```powershell
# 创建并激活conda环境
conda env create -f environment.yaml
conda activate pixel-aff

# 创建必要的目录
mkdir -Force data/oxford-iiit-pet data/cams output
```

**数据集准备：**
```powershell
# 解压数据集
tar -xzf images.tar.gz -C data/oxford-iiit-pet
tar -xzf annotations.tar.gz -C data/oxford-iiit-pet
```

**阶段1：训练分类网络并生成CAM**
```powershell
# 创建输出目录
mkdir -Force output/stage1_resnet50

# 运行第一阶段训练（使用CUDA优化）
python Task2_Weakly_Supersive_learning/run/run_train.py `
    --data_root ./data/oxford-iiit-pet `
    --stage 1 `
    --backbone resnet50 `
    --output_dir ./output `
    --cam_dir ./data/cams `
    --batch_size 16 `
    --lr 0.001 `
    --epochs 50 `
    --use_amp `
    --cudnn_benchmark `
    --empty_cache `
    --cuda_prefetch `
    --gpu_id 0
```

**阶段2：训练AffinityNet**
```powershell
# 创建第二阶段输出目录
mkdir -Force output/stage2_resnet50

# 运行第二阶段训练（使用CUDA优化）
python Task2_Weakly_Supersive_learning/run/run_train.py `
    --data_root ./data/oxford-iiit-pet `
    --stage 2 `
    --backbone resnet50 `
    --output_dir ./output `
    --cam_dir ./data/cams `
    --batch_size 16 `
    --lr 0.001 `
    --epochs 10 `
    --lambda_aff 0.1 `
    --model_path ./output/stage1_resnet50/model_best.pth.tar `
    --use_amp `
    --cudnn_benchmark `
    --empty_cache `
    --cuda_prefetch `
    --gpu_id 0
```

**阶段3：模型推理**
```powershell
# 创建推理输出目录
mkdir -Force output/inference

# 运行推理
python Task2_Weakly_Supersive_learning/run/run_inference.py `
    --data_root ./data/oxford-iiit-pet `
    --model_path ./output/stage2_resnet50/model_best.pth.tar `
    --output_dir ./output/inference `
    --backbone resnet50 `
    --split test `
    --num_iters 10 `
    --crf `
    --gpu_id 0 `
    --num_images 20
```

**阶段4：模型评估**
```powershell
# 创建评估输出目录
mkdir -Force output/eval_with_crf output/eval_no_crf

# 运行评估（带CRF后处理）
python Task2_Weakly_Supersive_learning/run/run_eval.py `
    --data_root ./data/oxford-iiit-pet `
    --model_path ./output/stage2_resnet50/model_best.pth.tar `
    --output_dir ./output/eval_with_crf `
    --split test `
    --num_iters 10 `
    --crf `
    --gpu_id 0 `
    --num_images 20

# 运行评估（不带CRF后处理）
python Task2_Weakly_Supersive_learning/run/run_eval.py `
    --data_root ./data/oxford-iiit-pet `
    --model_path ./output/stage2_resnet50/model_best.pth.tar `
    --output_dir ./output/eval_no_crf `
    --split test `
    --num_iters 10 `
    --gpu_id 0 `
    --num_images 20
```

**阶段5：消融实验**

**5.1 亲和力损失权重实验**
```powershell
# 以不同的lambda_aff值运行训练
python Task2_Weakly_Supersive_learning/run/run_train.py `
    --data_root ./data/oxford-iiit-pet `
    --stage 2 `
    --backbone resnet50 `
    --output_dir ./output `
    --cam_dir ./data/cams `
    --batch_size 8 `
    --lr 0.001 `
    --epochs 10 `
    --lambda_aff 0 `
    --model_path ./output/stage1_resnet50/model_best.pth.tar `
    --gpu_id 0

# 可以尝试其他值：0.01, 0.1, 1.0
```

**5.2 标签传播迭代次数实验**
```powershell
# 以不同的num_iters值运行评估
python Task2_Weakly_Supersive_learning/run/run_eval.py `
    --data_root ./data/oxford-iiit-pet `
    --model_path ./output/stage2_resnet50/model_best.pth.tar `
    --output_dir ./output/eval_iters_5 `
    --split test `
    --num_iters 5 `
    --crf `
    --gpu_id 0 `
    --num_images 20

# 可以尝试其他值：0, 10, 20
```

**5.4 骨干网络实验**
```powershell
# 使用ResNet101进行训练
python Task2_Weakly_Supersive_learning/run/run_train.py `
    --data_root ./data/oxford-iiit-pet `
    --stage 1 `
    --backbone resnet101 `
    --output_dir ./output `
    --cam_dir ./data/cams `
    --batch_size 8 `
    --lr 0.001 `
    --epochs 50 `
    --gpu_id 0
```

### 监控GPU使用情况

在训练期间可以使用以下命令监控GPU使用情况：

#### Linux/macOS:
```bash
# 在另一个终端窗口中运行，每秒更新一次
nvidia-smi -l 1
```

#### Windows (PowerShell):
```powershell
# 在另一个PowerShell窗口中运行，每秒更新一次
nvidia-smi -l 1
```

这将显示GPU温度、利用率、内存使用情况和功耗，帮助您确定最佳批次大小和优化参数。在内存不足时，可以尝试减小批次大小或启用更多CUDA优化选项。

### 批次大小选择指南

根据您的GPU显存大小，以下是推荐的批次大小设置：

| GPU显存 | 推荐批次大小（普通精度） | 推荐批次大小（混合精度 --use_amp） |
|---------|-------------------|-------------------------|
| 8GB    | 8-10              | 12-16                   |
| 12GB   | 12-16             | 20-24                   |
| 16GB+  | 16-24             | 24-32                   |

**注意**：实际可用批次大小会受到模型复杂度、输入图像尺寸和其他因素影响，可能需要根据您的具体硬件进行调整。

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

## CUDA训练加速

为了在单GPU环境下获得最佳性能，您可以使用以下CUDA优化参数：

### Bash (Linux/macOS)

```bash
# 启用所有CUDA优化选项的训练命令示例
python Task2_Weakly_Supersive_learning/run/run_train.py \
    --data_root ./data/oxford-iiit-pet \
    --stage 1 \
    --backbone resnet50 \
    --batch_size 16 \  # 根据您的GPU内存适当调整
    --output_dir ./output \
    --use_amp \       # 启用自动混合精度训练
    --cudnn_benchmark \ # 启用cudnn benchmark加速卷积操作
    --empty_cache \    # 每个epoch后清空CUDA缓存
    --cuda_prefetch \  # 预热CUDA缓存
    --gpu_id 0
```

### PowerShell (Windows)

```powershell
# 启用所有CUDA优化选项的训练命令示例
python Task2_Weakly_Supersive_learning/run/run_train.py `
    --data_root ./data/oxford-iiit-pet `
    --stage 1 `
    --backbone resnet50 `
    --batch_size 16 `  # 根据您的GPU内存适当调整
    --output_dir ./output `
    --use_amp `        # 启用自动混合精度训练
    --cudnn_benchmark ` # 启用cudnn benchmark加速卷积操作
    --empty_cache `     # 每个epoch后清空CUDA缓存
    --cuda_prefetch `   # 预热CUDA缓存
    --gpu_id 0
```

### CUDA优化选项说明

1. **自动混合精度训练 (`--use_amp`)**
   - 使用FP16和FP32混合精度进行模型训练
   - 可以显著提高训练速度（最多可提升30-50%）
   - 减少GPU内存使用，允许使用更大的批次大小

2. **CuDNN基准测试 (`--cudnn_benchmark`)**
   - 自动为当前配置寻找最佳卷积算法
   - 适用于模型架构和输入大小固定的情况
   - 第一次迭代有轻微延迟，后续迭代速度显著提升

3. **CUDA缓存优化 (`--empty_cache`)**
   - 在每个训练周期后释放未使用的CUDA内存
   - 减少内存碎片，避免训练过程中的内存不足错误
   - 推荐在GPU内存有限时使用

4. **CUDA缓存预热 (`--cuda_prefetch`)**
   - 在训练开始前预热CUDA缓存
   - 通过预先运行几次前向传播来优化内存分配和编译
   - 减少训练初期的波动

### 调整批次大小

为获得最佳性能，您可以调整批次大小(`--batch_size`)。一般规则：
- 更大的批次大小通常可以提高训练吞吐量
- 如果启用混合精度训练(`--use_amp`)，可以使用比正常情况更大的批次大小
- 根据您的GPU内存大小适当调整，避免出现OOM (Out of Memory)错误

### 监控GPU使用情况

在训练期间可以使用以下命令监控GPU使用情况：

```bash
# 在另一个终端窗口中运行,每10秒监控一次GPU状态
nvidia-smi -l 10
```

这将每秒更新一次GPU利用率和内存使用情况，帮助您确定最佳批次大小和优化参数。