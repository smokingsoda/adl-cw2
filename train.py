import datetime
import os.path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data.dataset import OxfordIIITPet
from eval import eval_classifier
from models.resnet import ResNet18

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2025)

dataset = OxfordIIITPet()
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                          persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                        persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                         persistent_workers=True)

resnet18 = ResNet18


def train_classifier(model):
    epochs = 5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()

        train_correct = 0.
        train_loss = 0.
        for x, y in train_loader:
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
        for x, y in val_loader:
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

    torch.save(model.state_dict(), "models/resnet34.pth")


def get_init_cam(model, x, y):
    model.to(device)
    model.eval()

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
    one_hot_y = F.one_hot(y, num_classes=37)
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

    cam = (cam > 0.5).float()

    return cam


if __name__ == '__main__':
    if not os.path.exists("models/resnet18.pth"):
        train_classifier(resnet18)
    else:
        resnet18.load_state_dict(torch.load("models/resnet18.pth"))
