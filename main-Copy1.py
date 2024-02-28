import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.class_balanced_loss import CB_loss
import timm
from tqdm import tqdm
from utils.models import *
from utils.dataset import *
from utils.metrics import *
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--path", default='', help="path to the data and save the model, figures to", type=str)
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=256, help="batch size used to train the model", type=int)
parser.add_argument("--num_classes", default=2, help="number of classes for the classifier", type=int)
parser.add_argument("--window_size", default=16, help="window size of the input data", type=int)
parser.add_argument("--label_type", default="hard", help="indicate whether use hard one-hot labels or soft numerical labels", choices=["hard", "soft"])
parser.add_argument("--long_tailed", default=0, help="indicate whether use balanced sampled data or the whole long-tailed data, 0 for balanced, 1 for long-tailed", type=int)
parser.add_argument("--exp_label", default=None, help="extra labels to distinguish between different experiments")
parser.add_argument("--cross_val_type", default=0, type=int, help="0 for train all val all, 1 for leave patient 1 out")
parser.add_argument("--task", default="regression", type=str, help="indicate what kind of task to be run (regression/classification, etc.)")
parser.add_argument("--normalize_roi", default=1, type=int, help="whether normalize the roi indices between [0, 1]")
parser.add_argument("--alpha", default=0.5, type=float, help="weight of cls loss, default 0.5")
args = parser.parse_args()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="rls",
    
    # track hyperparameters and run metadata
    config={
    "task": args.task,
    "window_size": args.window_size,
    "learning_rate": args.lr,
    "weight_decay": args.weight_decay,
    "batch_size": args.batch_size,
    "architecture": args.exp_label,
    "cross_val_type": args.cross_val_type,
    "normalize_roi": args.normalize_roi,
    "epochs": args.epochs,
    },
    
    # experiment name
    name=f"{args.task}_win{args.window_size}_epoch{args.epochs}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_cv{args.cross_val_type}_nr{args.normalize_roi}_{args.exp_label}"
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(3407)

# Hyperparameters
num_classes = args.num_classes  # Number of classes in ImageNet
window_size = args.window_size
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# load data
args.window_size = 'win' + str(args.window_size) + '_'
if args.long_tailed:
    args.window_size += 'LT_'

data_paths = ['data/12-13-2023', 'data/02-15-2024', 'data/02-17-2024']
if args.cross_val_type == 0:
    train_data = np.concatenate([np.load(os.path.join(path, args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32) for path in data_paths], axis=0)
    train_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_label_train.npy')) for path in data_paths], axis=0)
    train_roi_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_roi_label_train.npy')) for path in data_paths], axis=0)
    val_data = np.concatenate([np.load(os.path.join(path, args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32) for path in data_paths], axis=0)
    val_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_label_val.npy')) for path in data_paths], axis=0)
    val_roi_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_roi_label_val.npy')) for path in data_paths], axis=0)
else:
    leave_out_idx = args.cross_val_type - 1
    train_data = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
    train_label = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'EMG_label_train.npy')) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
    train_roi_label = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'EMG_roi_label_train.npy')) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
    val_data = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32), np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32)], axis=0)
    val_label = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_label_train.npy')), np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_label_val.npy'))], axis=0)
    val_roi_label = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_roi_label_train.npy')), np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_roi_label_val.npy'))], axis=0)

if args.label_type == "hard":
    positive_idx = train_label > 0
    train_label[positive_idx] = 1
    positive_idx = val_label > 0
    val_label[positive_idx] = 1
    train_label = train_label.astype(np.int_)
    val_label = val_label.astype(np.int_)
else:
    raise NotImplementedError("soft label not implemented yet!")
    
print(train_data.shape, train_data.dtype, train_label.shape, val_data.shape, val_label.shape, val_label.sum())

# original CNN transformation
transform = transforms.Compose([
    transforms.ToTensor(),
#     transforms.Resize(224),
#     transforms.RandomRotation(degrees=(-180, 180)),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize([0.47] * train_data.shape[1], [0.2] * train_data.shape[1]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
#     transforms.Resize(224),
#     transforms.RandomRotation(degrees=(-45, 45)),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize([0.47] * train_data.shape[1], [0.2] * train_data.shape[1]),
])

# ViT transformation
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# create dataset and dataloader
train_dataset = RLSDataset(train_data, train_label, train_roi_label, transform=transform, normalize_roi=args.normalize_roi)
val_dataset = RLSDataset(val_data, val_label, val_roi_label, transform=val_transform, normalize_roi=args.normalize_roi)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RLSRegressionModel(num_classes, window_size=window_size, pretrained=False)
model.to(device)

# Loss and optimizer
img_num_per_cls = np.unique(train_label, return_counts=True)[1]
# print(img_num_per_cls)
def CB_lossFunc(logits, labelList): #defince CB loss function
    return CB_loss(labelList, logits, img_num_per_cls, num_classes, "softmax", 0.9999, 2.0, device)
criterion = nn.CrossEntropyLoss()
L1Loss = nn.MSELoss()
# print('window size: ', args.window_size)
def regression_criterion(predict, roi_labels, labels):
    if not labels.sum():
        return (predict - predict).sum()
    mask = labels == 1
    return L1Loss(predict[mask], roi_labels[mask]) + (F.relu(predict[mask][:, 0] - predict[mask][:, 1].detach())).mean() + (F.relu(-predict[mask])).mean()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

alpha = 0.5
best_f1 = -np.inf
best_epoch = None
epsilon = 0.000001  # laplace smoothing

# Train the model
for epoch in range(num_epochs):
    running_cls_loss, running_regression_loss, total_samples = 0, 0, 0
    running_iou, total_iou_samples = 0, 0
    correct = 0
    matrix = np.zeros((2, 2))
    model.train()
    for i, (images, labels, roi_labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        roi_labels = roi_labels.to(device).float()
        
        # Forward pass
        cls_loss, regression_loss = 0, 0
        if args.task == "classification":
            cls_logits = model(images)
            cls_loss += criterion(outputs, labels)
        else:
            cls_logits, regression_logits = model(images)
            cls_loss += criterion(cls_logits, labels)
            regression_loss += regression_criterion(regression_logits, roi_labels, labels)
        loss = alpha * cls_loss + (1 - alpha) * regression_loss
        
        _, predicted = torch.max(cls_logits, 1)
        
        #compute the iou
        ious = calculate_iou(roi_labels, regression_logits.detach(), print_input=(epoch==num_epochs-1))
        running_iou += ious.sum()
        total_iou_samples += ious.size(0)
        #TODO: finish the case when no regression is conducted

        # Calculate the number of correct predictions
        correct += (predicted.detach() == labels).sum().item()
        
        # Calculate the number of correct predictions
        matrix[0][0] += (labels[predicted == 0] == 0).sum()
        matrix[0][1] += (labels[predicted == 1] == 0).sum()
        matrix[1][0] += (labels[predicted == 0] == 1).sum()
        matrix[1][1] += (labels[predicted == 1] == 1).sum()
        
        running_cls_loss += cls_loss.item()
        running_regression_loss += regression_loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Calculate accuracy
    total_samples = matrix.sum()
    TN, FP, FN, TP = matrix.ravel()
    overall_accuracy = ((TN + TP) / total_samples) * 100
    precision = TP / (TP + FP + epsilon) * 100
    recall = TP / (TP + FN + epsilon) * 100
    negative_recall = TN / (TN + FP + epsilon) * 100
    cls_balanced_accuracy = (recall + negative_recall) / 2
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    running_cls_loss /= total_samples
    running_regression_loss /= total_samples
    running_iou = running_iou * 100 / (total_iou_samples + epsilon)
    # Calculate accuracy
    accuracy = (correct / total_samples) * 100
    wandb.log({"train_cls_loss": running_cls_loss, "train_reg_loss": running_regression_loss, "train_f1": f1, "train_precision": precision, "train_recall": recall, "train_mIoU": running_iou})
    
    matrix = np.zeros((2, 2))
    correct = 0
    model.eval()
    running_cls_loss, running_regression_loss, total_samples = 0, 0, 0
    running_iou, total_iou_samples = 0, 0
    for i, (images, labels, roi_labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        roi_labels = roi_labels.to(device).float()

        # Forward pass
        cls_loss, regression_loss = 0, 0
        if args.task == 'classification':
            cls_logits = model(images)
            cls_loss += criterion(cls_logits, labels)
        else:
            cls_logits, regression_logits = model(images)
            mask = roi_labels[:, 0] != -1
            cls_loss += criterion(cls_logits, labels)
            regression_loss += regression_criterion(regression_logits, roi_labels, labels)
        loss = cls_loss + regression_loss
        running_cls_loss += cls_loss.item()
        running_regression_loss += regression_loss.item()
        
        _, predicted = torch.max(cls_logits, 1)
#         predicted = torch.ones(outputs.size(0))
#         predicted = torch.from_numpy(np.random.choice([0, 1], size=predicted.size(0)))

        #compute the iou
        ious = calculate_iou(roi_labels, regression_logits, print_input=(epoch==num_epochs-1))
        running_iou += ious.sum()
        total_iou_samples += ious.size(0)
        #TODO: finish the case when no regression is conducted
        
        # Calculate the number of correct predictions
        matrix[0][0] += (labels[predicted == 0] == 0).sum()
        matrix[0][1] += (labels[predicted == 1] == 0).sum()
        matrix[1][0] += (labels[predicted == 0] == 1).sum()
        matrix[1][1] += (labels[predicted == 1] == 1).sum()

    # Calculate accuracy
    total_samples = matrix.sum()
    TN, FP, FN, TP = matrix.ravel()
    overall_accuracy = ((TN + TP) / total_samples) * 100
    precision = TP / (TP + FP + epsilon) * 100
    recall = TP / (TP + FN + epsilon) * 100
    negative_recall = TN / (TN + FP + epsilon) * 100
    cls_balanced_accuracy = (recall + negative_recall) / 2
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    running_cls_loss /= total_samples
    running_regression_loss /= total_samples
    running_iou = running_iou * 100 / (total_iou_samples + epsilon)
    val_losses.append(running_cls_loss)
    val_accs.append(cls_balanced_accuracy)
    if f1 > best_f1:
        torch.save(model.state_dict(), os.path.join(args.path, 'best_model.ckpt'))
        best_f1 = f1
        best_epoch = epoch + 1
    wandb.log({"val_cls_loss": running_cls_loss, "val_reg_loss": running_regression_loss, "val_f1": f1, "val_precision": precision, "val_recall": recall, "val_mIoU": running_iou})
    print(f'Epoch [{epoch+1}/{num_epochs}], CLS Loss: {running_cls_loss:.4f}, regression Loss: {running_regression_loss:4f}, f1: {f1:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, mIoU: {running_iou:.2f}%')
    
wandb.finish()
print(f'The best f1 score is {best_f1:.2f} at epoch {best_epoch}')