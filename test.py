import torch
import os
from torch.utils.data import DataLoader, random_split
from data.dataset import OxfordIIITPet
from models.resnet import ResNet18
from models.unet import UNet
from utils.mask_utils import get_cam, get_trimap
from utils.loss import weighted_loss

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 检查数据和模型文件是否存在
print(f"数据文件夹存在: {os.path.exists('data')}")
print(f"模型文件存在: {os.path.exists('models/resnet18.pth')}")

# 加载数据集
try:
    dataset = OxfordIIITPet()
    print(f"数据集总大小: {len(dataset)}")

    # 检查图像文件夹是否有内容
    images_folder = "data/images/"
    image_count = len(
        [f for f in os.listdir(images_folder) if f.lower().endswith(".jpg")]
    )
    print(f"图像文件数量: {image_count}")

    # 计算各子集大小
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}, 测试集大小: {test_size}")

    # 设置随机种子确保可重复性
    torch.manual_seed(2025)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"随机划分后 - 测试集大小: {len(test_dataset)}")

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
    )
    print(f"测试加载器创建成功")
except Exception as e:
    print(f"数据集加载错误: {e}")
    raise


def test_classifier():
    print("测试分类器...")

    if len(test_dataset) == 0:
        print("错误: 测试数据集为空!")
        return

    try:
        model = ResNet18
        model.load_state_dict(
            torch.load(
                "models/resnet18.pth",
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        )
        model = model.to(device)
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载错误: {e}")
        return

    test_correct = 0
    total = 0

    try:
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                try:
                    if len(batch) != 3:
                        print(f"批次 {i} 数据格式错误, 长度为 {len(batch)}")
                        continue

                    x, y, _ = batch
                    batch_size = x.size(0)
                    print(f"处理批次 {i+1}, 样本数: {batch_size}")

                    if batch_size == 0:
                        print(f"批次 {i} 为空")
                        continue

                    x = x.to(device)
                    y = y.to(device)

                    outputs = model(x)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    test_correct += (predicted == y).sum().item()
                    print(f"已处理样本数: {total}, 正确预测数: {test_correct}")
                except Exception as e:
                    print(f"处理批次 {i} 时出错: {e}")
                    continue
    except Exception as e:
        print(f"测试循环错误: {e}")

    if total == 0:
        print("错误: 未处理任何样本!")
    else:
        accuracy = 100 * test_correct / total
        print(f"分类器测试集准确率: {accuracy:.2f}%")


def test_segmentation():
    print("\n测试分割模型...")
    if len(test_dataset) == 0:
        print("错误: 测试数据集为空!")
        return

    try:
        model = UNet()
        model.load_state_dict(
            torch.load(
                "models/unet.pth", map_location=torch.device("cpu"), weights_only=True
            )
        )
        model = model.to(device)
        model.eval()
        print("U-Net模型加载成功")
    except Exception as e:
        print(f"U-Net模型加载错误: {e}")
        return

    test_loss = 0.0
    test_iou = 0.0
    cam_iou = 0.0
    samples_processed = 0

    try:
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                try:
                    x, _, image_ids = batch
                    print(f"分割批次 {i+1}, 样本数: {len(x)}")

                    x = x.to(device)
                    trimap = get_trimap(image_ids).to(device)

                    pred_mask = model(x)
                    loss = weighted_loss(pred_mask, trimap)
                    test_loss += loss.item()

                    pred_binary = (pred_mask > 0.5).float()
                    intersection = (pred_binary * trimap).sum((1, 2, 3))
                    union = (pred_binary + trimap).clamp(0, 1).sum((1, 2, 3))
                    batch_iou = (intersection / (union + 1e-6)).sum().item()
                    test_iou += batch_iou

                    cam = get_cam(image_ids)
                    cam_binary = (cam > 0.5).float()
                    cam_intersection = (cam_binary * trimap).sum((1, 2, 3))
                    cam_union = (cam_binary + trimap).clamp(0, 1).sum((1, 2, 3))
                    cam_batch_iou = (cam_intersection / (cam_union + 1e-6)).sum().item()
                    cam_iou += cam_batch_iou

                    samples_processed += len(x)
                    print(f"已处理分割样本: {samples_processed}")
                except Exception as e:
                    print(f"分割批次 {i} 处理错误: {e}")

            if samples_processed > 0:
                test_loss /= samples_processed
                test_iou /= samples_processed
                cam_iou /= samples_processed

                print(f"分割模型测试结果:")
                print(f"测试损失: {test_loss:.4f}")
                print(f"测试IoU: {test_iou:.4f}")
                print(f"CAM IoU: {cam_iou:.4f}")
            else:
                print("错误: 分割测试未处理任何样本!")
    except Exception as e:
        print(f"分割测试循环错误: {e}")


if __name__ == "__main__":
    test_classifier()
    test_segmentation()
