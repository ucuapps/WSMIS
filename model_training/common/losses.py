import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """
    Implementation of mean soft-IoU loss for semantic segmentation
    """

    __EPSILON = 1e-6

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
        y_pred: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        y_true: a tensor of shape [B, H, W].
        Returns:
        float: soft-iou loss.
        """
        num_classes = y_pred.shape[1]

        y_true_dummy = torch.eye(num_classes)[y_true.squeeze(1)]
        y_true_dummy = y_true_dummy.permute(0, 3, 1, 2).to(y_true.device)

        y_pred_proba = F.softmax(y_pred, dim=1)

        intersection = torch.sum(y_pred_proba * y_true_dummy, dim=(2, 3))
        union = (
            torch.sum(y_pred_proba ** 2 + y_true_dummy ** 2, dim=(2, 3)) - intersection
        )
        iou_loss = (intersection + self.__EPSILON) / (union + self.__EPSILON)

        return 1 - iou_loss.mean()


class DiceLossWithLogits(nn.Module):
    @staticmethod
    def dice_metric(input, target):
        smooth = 1.0
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    def forward(self, input, target):
        input = (torch.sigmoid(input) > 0.5).type(torch.float32)
        return 1 - self.dice_metric(input, target)


class IALLoss(nn.Module):
    """Integral Attention Learning Loss"""

    EPSILON = 1e-8

    def __init__(self):
        super(IALLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: tensor of shape [B, C, H, W] with predicted logits
            y_true: tensor of shape [B, C, H, W] with soft probability labels
        Returns:
            (torch.tensor): 0-d torch tensor, representing the loss
        """
        scalar = torch.tensor([0]).float().cuda()
        pos = torch.gt(y_true, 0)
        neg = torch.eq(y_true, 0)
        pos_loss = -y_true[pos] * torch.log(torch.sigmoid(y_pred[pos]) + 1e-8)
        neg_loss = -torch.log(
            torch.exp(-torch.max(y_pred[neg], scalar.expand_as(y_pred[neg]))) + 1e-8
        ) + torch.log(1 + torch.exp(-torch.abs(y_pred[neg])))

        loss = 0.0
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        if num_pos > 0:
            loss += 1.0 / num_pos.float() * torch.sum(pos_loss)
        if num_neg > 0:
            loss += 1.0 / num_neg.float() * torch.sum(neg_loss)

        return loss


def get_loss(loss_config):
    loss_name = loss_config["name"]
    if loss_name == "categorical_cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=loss_config.get("ignore_index", -1))
    elif loss_name == "binary_cross_entropy":
        return nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor(loss_config.get("pos_weight"))
        )
    elif loss_name == "integral_attention_learning":
        return IALLoss()
    elif loss_name == "mean_iou":
        return IoULoss()
    elif loss_name == "kldiv":
        return nn.KLDivLoss(reduction="none")
    elif loss_name == "binary_dice":
        return DiceLossWithLogits()
    else:
        raise ValueError(f"Loss [{loss_name}] not recognized.")
