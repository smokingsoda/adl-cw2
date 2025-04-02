import datetime
import os.path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split

from data.dataset import OxfordIIITPet
from eval import eval_classifier
from models.resnet import ResNet18
from models.unet import UNet
from utils.loss import weighted_loss
from utils.mask_utils import create_cam, get_cam, get_trimap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2025)
batch_size = 32

dataset = OxfordIIITPet()
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

# In order to make the data in the CAM and batch match,
# the data set is disrupted in advance and the Dataloader is loaded without disrupting the train_loader.
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                          persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                        persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                         persistent_workers=True)

resnet = ResNet18
unet = UNet()


def train_classifier(model):
    epochs = 10
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(epochs):
        model.train()

        train_correct = 0.
        train_loss = 0.
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_pred, 1)
            train_correct += (predicted == y).sum().item()
            train_loss += loss.item() * x.shape[0]

        train_acc = train_correct / len(train_dataset)
        train_loss /= len(train_dataset)

        # evaluation on val dataset
        model.eval()
        val_correct = 0.
        val_loss = 0.
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)

            val_batch_correct, val_batch_loss = eval_classifier(model, x, y)
            val_correct += val_batch_correct
            val_loss += val_batch_loss * x.shape[0]

        val_acc = val_correct / len(val_dataset)
        val_loss /= len(val_dataset)

        # if epoch % 10 == 0:
        print(f"EPOCH: {epoch + 1}/{epochs}, train_loss: {train_loss}, train_acc: {train_acc * 100:.2f}% "
              f"val_loss: {val_loss}, val_acc: {val_acc * 100:.2f}%, {datetime.datetime.now()}")

    torch.save(model.state_dict(), "models/resnet18.pth")


def train_unet(model):
    print(f"start training UNet, {datetime.datetime.now()}")

    epochs = 10
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = 0.
        for i, (x, _, image_ids) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            cam = get_cam(image_ids)

            if epoch == 0:
                mask = cam
            elif epoch < 3:
                mask = 0.8 * cam + 0.2 * model(x).detach()
            else:
                alpha = min(0.3 * (epoch - 3), 0.6)
                mask = (1 - alpha) * cam + alpha * model(x).detach()

            mask = torch.clamp(mask, 0, 1)

            pred_mask = model(x)
            loss = weighted_loss(pred_mask, mask)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataset)

        # evaluation
        model.eval()
        val_loss = 0.
        val_iou = 0.
        for x, _, image_ids in val_loader:
            x = x.to(device)
            trimap = get_trimap(image_ids).to(device)

            with torch.no_grad():
                pred_mask = model(x)
                loss = weighted_loss(pred_mask, trimap)
                val_loss += loss.item() * x.shape[0]

                pred_binary = (pred_mask > 0.5).float()
                intersection = (pred_binary * trimap).sum((1, 2, 3))
                union = (pred_binary + trimap).clamp(0, 1).sum((1, 2, 3))
                batch_iou = (intersection / (union + 1e-6)).mean().item()
                val_iou += batch_iou * x.shape[0]

        val_loss /= len(val_dataset)
        val_iou /= len(val_dataset)

        print(f"EPOCH: {epoch + 1}/{epochs}, train_loss: {train_loss}, val_loss: {val_loss}, "
              f"val_iou:{val_iou}, {datetime.datetime.now()}")

    torch.save(model.state_dict(), "models/unet.pth")


# 1. 逆转归一化
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将归一化的Tensor逆转回原始值域"""
    # 深拷贝避免修改原Tensor
    tensor = tensor.clone()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor.mul_(std).add_(mean)  # 逆运算：x = (x_norm * std) + mean
    return tensor.clamp_(0, 1)  # 裁剪到[0,1]范围


if __name__ == '__main__':
    if not os.path.exists("models/resnet18.pth"):
        train_classifier(resnet)
    else:
        resnet.load_state_dict(torch.load("models/resnet18.pth"))

    if not os.path.exists("data/CAM"):
        for loader in [train_loader, val_loader, test_loader]:
            for i, (x, y, ids) in enumerate(loader):
                create_cam(resnet, x, y, ids)

    # clear cache to provide more space for training UNet
    resnet = resnet.to('cpu')
    torch.cuda.empty_cache()

    if not os.path.exists("models/unet.pth"):
        train_unet(unet)
    else:
        unet.load_state_dict(torch.load("models/unet.pth"))
    unet.eval()
    unet = unet.to(device)

    test_loss = 0.
    test_iou = 0.

    # test
    for x, _, trimap in test_loader:
        x = x.to(device)
        trimap = trimap.to(device)

        with torch.no_grad():
            pred_mask = unet(x)
            loss = weighted_loss(pred_mask, trimap)
            test_loss += loss.item() * x.shape[0]

            pred_binary = (pred_mask > 0.5).float()
            intersection = (pred_binary * trimap).sum((1, 2, 3))
            union = (pred_binary + trimap).clamp(0, 1).sum((1, 2, 3))
            batch_iou = (intersection / (union + 1e-6)).mean().item()
            test_iou += batch_iou * x.shape[0]

    test_loss /= len(val_dataset)
    test_iou /= len(val_dataset)

    print(f"test_loss: {test_loss},test_iou:{test_iou}, {datetime.datetime.now()}")

    # display samples
    unet.eval()
    for i, (x, y, trimap) in enumerate(test_loader):
        x = x.to(device)
        trimap = trimap.to(device)

        x_denorm = denormalize(x[0].unsqueeze(0).to('cpu'))  # 保持batch维度处理
        image_np = x_denorm.squeeze(0).permute(1, 2, 0).numpy()  # C×H×W → H×W×C
        image_uint8 = (image_np * 255).astype(np.uint8)  # 转为0-255整型

        pred_pil = Image.fromarray(image_uint8)
        pred_pil.show()

        binary_image = (trimap[0].squeeze(0) > 0.5).float() * 255
        pil_image = Image.fromarray(binary_image.to('cpu').numpy().astype(np.uint8), mode='L')
        pil_image.show()

        with torch.no_grad():
            pred_mask = unet(x)

        binary_image = (pred_mask[0].squeeze(0) > 0.5).float() * 255
        pil_image = Image.fromarray(binary_image.to('cpu').numpy().astype(np.uint8), mode='L')
        pil_image.show()

        cam = torch.load('data/CAM/cam_0.pt')
        binary_image = (cam[0].squeeze(0) > 0.5).float() * 255
        pil_image = Image.fromarray(binary_image.to('cpu').numpy().astype(np.uint8), mode='L')
        pil_image.show()

        break
