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
from utils.lossfunc import *
import wandb
from train import *

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
args.device = device

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
train_transform = get_cnn_transforms(train_data.shape[1])
val_transform = get_cnn_transforms(train_data.shape[1], train=False)

# create dataset and dataloader
train_dataset = RLSDataset(train_data, train_label, train_roi_label, transform=train_transform, normalize_roi=args.normalize_roi)
val_dataset = RLSDataset(val_data, val_label, val_roi_label, transform=val_transform, normalize_roi=args.normalize_roi)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RLSRegressionModel(num_classes, window_size=window_size, pretrained=False)
model.to(device)

# Loss and optimizer
img_num_per_cls = np.unique(train_label, return_counts=True)[1]

cls_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
args.best_f1 = -np.inf
args.best_epoch = None

# main train val loop
for epoch in range(num_epochs):
    args.epoch = epoch
    train(args, model, train_loader, cls_criterion, regression_criterion, optimizer, scheduler)
    val(args, model, val_loader, cls_criterion, regression_criterion)
    
wandb.finish()
print(f'The best f1 score is {args.best_f1:.2f} at epoch {args.best_epoch}')

#TODO: check wandb log to see if it is expected, debug the transforms, the normalize transform seems not to do expected thing