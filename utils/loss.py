import torch
import torch.nn.functional as F


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