import datetime
import os.path

import torch
from torch.utils.data import DataLoader, random_split

from models.resnet import ResNet18
from data.dataset import OxfordIIITPet
from eval import eval_classifier

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


def train_resnet():
    epochs = 10
    optimizer = torch.optim.Adam(params=resnet18.parameters(), lr=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()

    resnet18.to(device)

    for epoch in range(epochs):
        resnet18.train()

        train_correct = 0.
        train_loss = 0.
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = resnet18(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_pred, 1)
            train_correct += (predicted == y).sum().item()
            train_loss += loss.item() * x.shape[0]

        train_acc = train_correct / len(train_dataset)
        train_loss /= len(train_dataset)

        # evaluation on val dataset
        resnet18.eval()
        val_correct = 0.
        val_loss = 0.
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            val_batch_correct, val_batch_loss = eval_classifier(resnet18, x, y)
            val_correct += val_batch_correct
            val_loss += val_batch_loss * x.shape[0]

        val_acc = val_correct / len(val_dataset)
        val_loss /= len(val_dataset)

        # if epoch % 10 == 0:
        print(f"EPOCH: {epoch + 1}/{epochs}, train_loss: {train_loss}, train_acc: {train_acc * 100:.2f}% "
              f"val_loss: {val_loss}, val_acc: {val_acc * 100:.2f}%, {datetime.datetime.now()}")

    torch.save(resnet18.state_dict(), "models/resnet18.pth")


if __name__ == '__main__':
    if not os.path.exists("models/resnet18.pth"):
        train_resnet()
    else:
        resnet18.load_state_dict(torch.load("models/resnet18.pth"))
