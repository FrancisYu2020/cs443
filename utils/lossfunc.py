import torch.nn as nn
import torch.nn.functional as F

def CB_lossFunc(logits, labelList): #defince CB loss function
    return CB_loss(labelList, logits, img_num_per_cls, num_classes, "softmax", 0.9999, 2.0, device)

def regression_criterion(predict, roi_labels, labels):
    if not labels.sum():
        return (predict - predict).sum()
    mask = labels == 1
    return F.mse_loss(predict[mask], roi_labels[mask]) + (F.relu(predict[mask][:, 0] - predict[mask][:, 1].detach())).mean() + (F.relu(-predict[mask])).mean()