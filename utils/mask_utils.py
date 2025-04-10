import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# save grad_cam in local as .pt file
def create_cam(model, x, y, image_ids):
    os.makedirs("data/CAM", exist_ok=True)
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
    cam = torch.einsum("bc,bchw->bhw", pooled_gradients, features)

    # block negative values
    cam = F.relu(cam)

    # Normalization
    cam = (cam - cam.amin(dim=(1, 2), keepdim=True)[0]) / (
        cam.amax(dim=(1, 2), keepdim=True)[0] + 1e-8
    )
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(x.shape[2], x.shape[3]),
        mode="bilinear",
        align_corners=False,
    )

    for i in range(x.shape[0]):
        single_cam = cam[i, 0].detach().cpu()  # (H,W)
        torch.save(single_cam, f"data/CAM/{image_ids[i]}.pt")


def get_cam(image_ids):
    cam = [
        torch.load(f"data/CAM/{image_id}.pt", map_location=device, weights_only=False)
        .reshape(1, 1, 256, 256)
        .to(device)
        for image_id in image_ids
    ]
    return torch.cat(cam, dim=0)


def get_trimap(image_ids):
    trimaps = []
    for image_id in image_ids:
        trimap_path = f"data/annotations/trimaps/{image_id}.png"
        trimap = Image.open(trimap_path)
        trimap = transforms.Resize((256, 256))(trimap)

        trimap = np.array(trimap)
        trimap[trimap == 2] = 0
        trimap[trimap == 3] = 1
        trimap = torch.from_numpy(trimap).float().reshape(1, 1, 256, 256)

        trimaps.append(trimap)
    return torch.cat(trimaps, dim=0)
