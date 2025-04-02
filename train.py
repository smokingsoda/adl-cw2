import datetime
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data.dataset import OxfordIIITPet
from eval import eval_classifier
from models.resnet import ResNet18
from models.unet import UNet
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_softmax

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2025)
batch_size = 32

dataset = OxfordIIITPet()
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

# In order to make the data in the CAM and batch match,
# the data set is disrupted in advance and the Dataloader is loaded without disrupting the train_loader.
indices = np.random.permutation(len(dataset))
shuffled_dataset = torch.utils.data.Subset(dataset, indices)
train_dataset, val_dataset, test_dataset = random_split(shuffled_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
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


# grad_cam
def create_cam(model, x, y, batch):
    model = model.to(device)
    model.eval()
    x = x.to(device)
    y = y.to(device)

    features = []
    gradients = []

    # hook method to keep feature map
    def hook_feature(module, input, output):
        features.append(output)

    # hook method to keep gradient map
    def hook_grad(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # handles to hook feature and grad in forward and backward process
    handle = model.layer4[-1].conv2.register_forward_hook(hook_feature)
    handle_grad = model.layer4[-1].conv2.register_full_backward_hook(hook_grad)
    logits = model(x)

    model.zero_grad()  # set gradient to 0

    # only the gradient of the target class for network parameters is calculated.
    one_hot_y = F.one_hot(y, num_classes=37).to(device)
    logits.backward(gradient=one_hot_y, retain_graph=True)

    # remove hook
    handle.remove()
    handle_grad.remove()

    # Only use the data hooked from the first invoke
    features = features[0]  # shape = (B,C,H,W)
    gradients = gradients[0]  # shape = (B,C,H,W)

    # GAP layer
    pooled_gradients = torch.mean(gradients, dim=[2, 3])  # shape = (B,C)

    # weighted sum of channels
    cam = torch.einsum('bc,bchw->bhw', pooled_gradients, features)

    # block negative values
    cam = F.relu(cam)

    # Normalization
    cam = (cam - cam.amin(dim=(1, 2), keepdim=True)[0]) / (cam.amax(dim=(1, 2), keepdim=True)[0] + 1e-8)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(x.shape[2], x.shape[3]),
        mode='bilinear',
        align_corners=False
    )

    # normalize
    cam = (cam - cam.amin(dim=(1, 2), keepdim=True)) / (
            cam.amax(dim=(1, 2), keepdim=True) - cam.amin(dim=(1, 2), keepdim=True) + 1e-8)

    # add gaussian noise to smooth
    cam = F.conv2d(cam,
                   torch.ones(1, 1, 3, 3, device=device) / 9.0,
                   padding=1)

    # CRF Processing (per image in batch)
    refined_cams = []
    for i in range(x.shape[0]):
        img_np = x[i].mul(255).byte().cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
        cam_np = cam[i, 0].detach().cpu().numpy()  # (H,W)

        img_np = np.ascontiguousarray(img_np)
        cam_np = np.ascontiguousarray(cam_np)

        # Skip CRF if image is invalid
        if np.max(cam_np) < 0.1:
            refined_cams.append(cam[i])
            continue

        # CRF Parameters
        d = densecrf.DenseCRF2D(img_np.shape[1], img_np.shape[0], 2)  # 2 classes

        # Unary potential (negative log probability)
        U = unary_from_softmax(np.stack([1 - cam_np, cam_np], axis=0))
        d.setUnaryEnergy(U)

        # Pairwise potentials
        d.addPairwiseGaussian(sxy=3, compat=3)  # Spatial
        d.addPairwiseBilateral(
            sxy=10,  # Spatial radius
            srgb=13,  # Color radius
            rgbim=img_np,
            compat=10  # Weight
        )

        # Inference
        Q = d.inference(5)  # 5 iterations
        refined = np.argmax(Q, axis=0).reshape(cam_np.shape)
        refined_cams.append(torch.from_numpy(refined).float().to(device))

    # Stack results
    final_cam = torch.stack(refined_cams).unsqueeze(1)  # (B,1,H,W)
    final_cam = (final_cam - final_cam.min()) / (final_cam.max() - final_cam.min() + 1e-8)

    torch.save(final_cam, f"data/CAM/cam_{batch}.pt")


def weighted_loss(pred, target, lambda_dice=0.5, lambda_iou=0.3):
    # BCE loss with label smooth
    bce_loss = F.binary_cross_entropy(
        pred,
        target * 0.9 + 0.05,  # label smooth
        reduction='mean'
    )

    # Dice loss
    intersection = (pred * target).sum()
    dice_loss = 1 - (2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)

    # IoU loss
    union = (pred + target).clamp(0, 1).sum()
    iou_loss = 1 - (intersection + 1e-5) / (union + 1e-5)

    # edge loss
    grad_x_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    grad_y_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    edge_loss = (grad_x_pred.mean() + grad_y_pred.mean()) * 0.5

    return (
            (1 - lambda_dice - lambda_iou) * bce_loss +
            lambda_dice * dice_loss +
            lambda_iou * iou_loss +
            0.1 * edge_loss
    )


def train_unet(model):
    print(f"start training UNet, {datetime.datetime.now()}")

    epochs = 10
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = 0.
        for i, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()

            cam = torch.load(f"data/CAM/cam_{i}.pt")

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
        for x, _, trimap in val_loader:
            x = x.to(device)
            trimap = trimap.to(device)

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


if __name__ == '__main__':
    if not os.path.exists("models/resnet18.pth"):
        train_classifier(resnet)
    else:
        resnet.load_state_dict(torch.load("models/resnet18.pth"))

    if not os.path.exists("data/CAM"):
        os.makedirs("data/CAM", exist_ok=True)
        for i, (x, y, _) in enumerate(train_loader):
            create_cam(resnet, x, y, i)

    # clear cache to provide more space for training UNet
    resnet = resnet.to('cpu')
    torch.cuda.empty_cache()

    unet.load_state_dict(torch.load("models/unet.pth"))
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
