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

## 实验与使用方法

本节介绍了如何使用项目代码进行弱监督语义分割的训练、推理与评估实验。

### 准备工作

在开始实验之前，请完成以下准备工作：

```bash
# 创建并激活conda环境
conda env create -f environment.yaml
conda activate pixel-aff

# 或者使用pip安装依赖
# pip install -r requirements.txt

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

### 阶段 1: 训练分类网络并生成 CAM

此阶段训练一个基于图像级标签的分类器，并生成后续阶段所需的类激活图 (CAM)。

```bash
# 创建输出目录
mkdir -p output/stage1_resnet50

# 运行第一阶段训练
python Task2_Weakly_Supersive_learning/run/run_train.py \
    --data_root ./data/oxford-iiit-pet \
    --stage 1 \
    --backbone resnet50 \
    --output_dir ./output \
    --cam_dir ./data/cams \
    --batch_size 8 \
    --lr 0.001 \
    --epochs 50 \
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

# 运行第二阶段训练
python Task2_Weakly_Supersive_learning/run/run_train.py \
    --data_root ./data/oxford-iiit-pet \
    --stage 2 \
    --backbone resnet50 \
    --output_dir ./output \
    --cam_dir ./data/cams \
    --batch_size 8 \
    --lr 0.001 \
    --epochs 10 \
    --lambda_aff 0.1 \
    --model_path ./output/stage1_resnet50/model_best.pth.tar \
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
    --backbone resnet50 \
    --split test \
    --num_iters 10 \
    --crf \
    --gpu_id 0

# 运行评估（不带CRF后处理）
python Task2_Weakly_Supersive_learning/run/run_eval.py \
    --data_root ./data/oxford-iiit-pet \
    --model_path ./output/stage2_resnet50/model_best.pth.tar \
    --output_dir ./output/eval_no_crf \
    --backbone resnet50 \
    --split test \
    --num_iters 10 \
    --gpu_id 0
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
    --lambda_aff 0 \  # 尝试不同值：0, 0.01, 0.1, 1.0
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
    --output_dir ./output/eval_iters_X \  # X替换为具体的迭代数值
    --backbone resnet50 \
    --split test \
    --num_iters 5 \  # 尝试不同值：0, 5, 10, 20
    --crf \
    --gpu_id 0
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

### 不同操作系统的运行说明

#### Windows (PowerShell)

在Windows系统中，请将上述bash命令中的反斜杠 `\` 替换为重音符 `` ` ``，例如：

```powershell
python Task2_Weakly_Supersive_learning/run/run_train.py `
    --data_root ./data/oxford-iiit-pet `
    --stage 1 `
    --backbone resnet50 `
    # ... 其他参数
```

#### macOS/Linux

macOS和Linux系统可以直接使用上述bash命令。

## 实现细节
1. 使用Oxford-IIIT宠物数据集（37个类别）进行训练和评估
2. 实现了一个双阶段训练过程：
   - 第一阶段：使用图像级别的标签训练分类器，生成CAM
   - 第二阶段：利用CAM生成亲和力标签，训练亲和力模型
3. 支持不同的骨干网络（ResNet50、ResNet101）
4. 实现了标签传播算法，利用学习的亲和力改进分割
5. 支持CRF后处理以进一步优化分割边界

## 实验 (Experiments)

本节详细说明了根据项目需求文档 (COMP0197) 进行实验的建议步骤，以评估实现的弱监督分割框架。

### 阶段 0: 准备工作

在开始实验之前，请确保以下准备工作已完成：

1.  **环境激活:** 激活 Conda 环境:
    ```bash
    conda activate pixel-aff
    ```
2.  **数据集:** 确认 Oxford-IIIT Pet 数据集已下载并解压到 `--data_root` 参数指定的路径 (默认为 `./data/oxford-iiit-pet`)。
3.  **目录创建:** 创建必要的输出和中间文件目录：
    ```bash
    mkdir -p ./output ./data/cams
    ```
4.  **代码与配置检查:**
    *   **README 参数:** 仔细检查并根据您的最终代码实现更新本 README 文件中的参数说明（特别是学习率、epoch 数等）。
    *   **伪标签生成:** **(关键)** 确认 `utils/datasets.py` 中的 `AffinityPetDataset._generate_affinity_mask` 函数严格按照论文思想，在**低分辨率 (56x56)** 下，利用 CAM 生成**伪亲和力标签**。特别注意置信区域和中性区域的处理是否符合预期。

### 阶段 1: 训练分类网络并生成 CAM (`--stage 1`)

此阶段训练一个基于图像级标签的分类器，并生成后续阶段所需的类激活图 (CAM)。

1.  **运行训练:**
    ```bash
    python run/run_train.py \
        --data_root ./data/oxford-iiit-pet \
        --stage 1 \
        --backbone resnet50 \
        --output_dir ./output \
        --cam_dir ./data/cams \
        --batch_size 8 \
        --lr 0.001 \
        --epochs 50 \
        --gpu_id 0 
        # 根据需要调整 backbone, batch_size, lr, epochs, gpu_id
        # 50个epoch和lr=0.001是常见的起点，可根据实际情况调整
    ```
2.  **预期输出:**
    *   训练好的分类模型权重，保存在 `./output/stage1_resnet50/model_best.pth.tar` (路径可能因backbone调整)。
    *   为训练集和验证集生成的 CAM 文件 (`.npy` 格式)，保存在 `./data/cams` 目录下。

### 阶段 2: 训练 AffinityNet (`--stage 2`)

此阶段利用阶段 1 生成的 CAM（处理为伪亲和力标签）来训练 AffinityNet。

1.  **运行训练:**
    ```bash
    python run/run_train.py \
        --data_root ./data/oxford-iiit-pet \
        --stage 2 \
        --backbone resnet50 \
        --output_dir ./output \
        --cam_dir ./data/cams \
        --batch_size 8 \
        --lr 0.001 \
        --epochs 10 \
        --lambda_aff 0.1 \
        --model_path ./output/stage1_resnet50/model_best.pth.tar \
        --gpu_id 0
        # 确保 --cam_dir 指向阶段1生成的CAM
        # 确保 --model_path 指向阶段1训练好的模型权重 (用于加载骨干网络)
        # 使用您确定的超参数 (epochs=10, lambda_aff=0.1)
        # 注意学习率设置: 骨干网络lr=args.lr*0.1, 其他层lr=args.lr
    ```
2.  **预期输出:**
    *   训练好的 AffinityNet 模型权重，保存在 `./output/stage2_resnet50/model_best.pth.tar` (路径可能因backbone调整)。

### 阶段 3: 评估弱监督模型性能

使用训练好的 AffinityNet (结合标签传播) 在测试集上评估分割性能。

1.  **运行评估 (不使用 CRF):**
    ```bash
    python run/run_eval.py \
        --data_root ./data/oxford-iiit-pet \
        --model_path ./output/stage2_resnet50/model_best.pth.tar \
        --output_dir ./output/eval_no_crf \
        --backbone resnet50 \
        --split test \
        --gpu_id 0
    ```
2.  **运行评估 (使用 CRF):**
    ```bash
    python run/run_eval.py \
        --data_root ./data/oxford-iiit-pet \
        --model_path ./output/stage2_resnet50/model_best.pth.tar \
        --output_dir ./output/eval_with_crf \
        --backbone resnet50 \
        --split test \
        --crf \
        --gpu_id 0
    ```
3.  **预期输出:**
    *   在各自的 `--output_dir` 中生成：
        *   `results.npy`: 包含 Mean Precision, Recall, F1, IoU 等数值结果。
        *   `metrics.png`: 指标的可视化图表。
        *   每个测试图像的评估可视化结果 (`*_eval.png`)。

### 阶段 4: 与全监督基线比较 (MRP 要求)

1.  **获取基线结果:** 与负责全监督部分的同事协作，获取使用相同骨干网络 (e.g., ResNet50) 和 Oxford Pets 数据集像素级标签训练的模型的评估结果 (特别是 Mean IoU)。
2.  **对比分析:** 在您的个人报告中，将阶段 3 得到的弱监督模型性能（建议使用 CRF 后的结果）与全监督基线进行量化比较。讨论性能差距、潜在原因以及弱监督在标注成本上的优势。

### 阶段 5: 消融实验 (MRP 要求)

为了深入理解框架各组件和超参数的影响，需要进行以下消融研究。对于每个研究，通常需要重新运行**阶段 2 (AffinityNet 训练)** 和**阶段 3 (评估)**。

1.  **亲和力损失权重 (`--lambda_aff`):**
    *   **目的:** 验证亲和力学习的重要性。
    *   **实验:** 尝试不同的 `--lambda_aff` 值进行 Stage 2 训练，例如 `0`, `0.01`, `0.1` (默认), `1.0`。然后进行 Stage 3 评估。比较 `lambda_aff=0` (无亲和力损失) 与其他值的性能差异。
2.  **标签传播迭代次数 (`--num_iters` in `eval.py`):**
    *   **目的:** 研究标签传播步骤的效果及迭代次数影响。
    *   **实验:** 使用训练好的最佳 AffinityNet 模型 (来自默认设置的 Stage 2)，在运行 `run/run_eval.py` 时，改变 `--num_iters` 参数，例如 `0` (无传播), `5`, `10` (默认), `20`。比较评估结果。
3.  **CRF 后处理 (`--crf` in `eval.py`):**
    *   **目的:** 量化 CRF 对分割边界精度的提升效果。
    *   **实验:** 比较阶段 3 中开启 (`--crf`) 和关闭 CRF 时的评估结果。
4.  **(可选) 主干网络 (`--backbone`):**
    *   **目的:** 探究更强大主干网络的影响。
    *   **实验:** 如果资源允许，将 `--backbone` 设为 `resnet101`，重新运行 Stage 1, Stage 2, Stage 3，并与 `resnet50` 的结果比较。
5.  **(可选) CAM 阈值 (`threshold` in `AffinityPetDataset`):**
    *   **目的:** 研究伪亲和力标签生成对 CAM 阈值的敏感性。
    *   **实验:** 修改 `utils/datasets.py` 中 `AffinityPetDataset` 的 `self.threshold` 值 (例如 0.2, 0.3, 0.4)，重新运行 Stage 2 训练和 Stage 3 评估。

**记录:** 对所有实验，请系统地记录使用的参数配置和得到的评估结果，以便在报告中进行分析和展示。

## 引用

```
Ahn, J., & Kwak, S. (2018). Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4981-4990).
```

Oxford-IIIT Pet Dataset:
```
Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
```