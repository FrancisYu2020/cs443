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
from train import *
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.01, help="learning rate used to train the model", type=float)
parser.add_argument("--weight_decay", default=0.1, help="weight decay used to train the model", type=float)
parser.add_argument("--epochs", default=300, help="epochs used to train the model", type=int)
parser.add_argument("--batch_size", default=256, help="batch size used to train the model", type=int)
parser.add_argument("--num_classes", default=2, help="number of classes for the classifier", type=int)
parser.add_argument("--window_size", default=16, help="window size of the input data", type=int)
parser.add_argument("--cross_val_type", default=0, type=int, help="0 for train all val all, 1 for leave patient 1 out")
parser.add_argument("--task", default="regression", type=str, help="indicate what kind of task to be run (regression/classification, etc.)")
parser.add_argument("--normalize_roi", default=1, type=int, help="whether normalize the roi indices between [0, 1]")
parser.add_argument("--alpha", default=0.5, type=float, help="weight of cls loss, default 0.5")
parser.add_argument("--architecture", default="3d-resnet18", choices=["3d-resnet10", "3d-resnet18", "2d-resnet18", "ViT-tiny"], help="architecture used")
args = parser.parse_args()

args.exp_name = f"{args.task}_win{args.window_size}_epoch{args.epochs}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_cv{args.cross_val_type}_nr{args.normalize_roi}_{args.architecture}"

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

data_paths = ['data/12-13-2023', 'data/02-15-2024', 'data/02-17-2024']
# if args.cross_val_type == 0:
#     val_data = np.concatenate([np.load(os.path.join(path, args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32) for path in data_paths], axis=0)
#     val_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_label_val.npy')) for path in data_paths], axis=0)
#     val_roi_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_roi_label_val.npy')) for path in data_paths], axis=0)
# else:
if 1:
#     leave_out_idx = args.cross_val_type - 1
    leave_out_idx = 0
    val_data = np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32)
    val_label = np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_label_val.npy'))
    val_roi_label = np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_roi_label_val.npy'))

positive_idx = val_label > 0
val_label[positive_idx] = 1
val_label = val_label.astype(np.int_)
    
print(val_data.shape, val_label.shape, val_label.sum())

# original CNN transformation
val_transform = get_cnn_transforms(val_data.shape[1], train=False)

# create dataset and dataloader
val_dataset = RLSDataset(args.architecture, val_data, val_label, val_roi_label, transform=val_transform, normalize_roi=args.normalize_roi)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            
# main eval code
def val(args, model, val_loader, metric, height=0.3, linewidth=1, offset=-20000, yoffset=0.35, fontsize=15, figsize=(10, 6)):
    '''
    args: input arguments from main function
    model: model to be evaluated 
    train_loader: training data dataloader
    epsilon: the laplace smoothing factor used to prevent division by 0
    cls_criterion: classification loss criterion
    regression_criterion: regression loss criterion
    '''
    plt.figure(figsize=figsize)
    
    def plot_regression(roi_labels, y_position):
        '''
        sub function to plot regression part
        '''
        if roi_labels[i][1] < 0:
            rect2 = Rectangle((curr_start, y_position), window_size, height, edgecolor='black', facecolor='black', linewidth=linewidth)
            plt.gca().add_patch(rect2)
        else:
            if roi_labels[i][0] < 0:
                roi_labels[i][0] = 0
            if roi_labels[i][0]:
                rect2 = Rectangle((curr_start, y_position), roi_labels[i][0], height, edgecolor='black', facecolor='black', linewidth=linewidth)
                plt.gca().add_patch(rect2)
            rect2 = Rectangle((roi_labels[i][0] + curr_start, y_position), roi_labels[i][-1] - roi_labels[i][0], height, edgecolor='red', facecolor='red', linewidth=linewidth)
            plt.gca().add_patch(rect2)
            if roi_labels[i][-1] < window_size - 1:
                rect2 = Rectangle((roi_labels[i][-1] + curr_start, y_position), window_size - 1 - roi_labels[i][-1], height, edgecolor='black', facecolor='black', linewidth=linewidth)
                plt.gca().add_patch(rect2)
            
    curr_start = 0
    for images, labels, roi_labels in tqdm(val_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        roi_labels = roi_labels.to(args.device).float()
        if args.normalize_roi:
            roi_labels = (roi_labels * window_size).int()

        # Forward pass
        cls_loss, regression_loss = 0, 0
        if args.task == 'classification':
            cls_logits = model(images)
        else:
            cls_logits, regression_logits = model(images)
            mask = roi_labels[:, 0] != -1
            if args.normalize_roi:
                regression_logits = (regression_logits * window_size).int().cpu()
            else:
                regression_logits = regression_logits.int().cpu()
        
        _, predicted = torch.max(cls_logits, 1)
        for i in range(len(predicted)):
            roi_labels = roi_labels.cpu()
            
            # classification ground truth 
            rect0 = Rectangle((curr_start, 0), window_size, height, edgecolor='red' if labels[i] else 'black', facecolor='red' if labels[i] else 'black', linewidth=linewidth)
            plt.gca().add_patch(rect0)
            
            # classification prediction
            rect1 = Rectangle((curr_start, 1), window_size, height, edgecolor='red' if predicted[i] else 'black', facecolor='red' if predicted[i] else 'black', linewidth=linewidth)
            plt.gca().add_patch(rect1)
            
            # regression ground truth
            plot_regression(roi_labels, 2)
            
            # regression prediction
            plot_regression(regression_logits, 3)
                    
            curr_start += window_size
        
    plt.text(curr_start + offset, 0 + yoffset, 'classification ground truth', fontsize=fontsize)
    plt.text(curr_start + offset, 1 + yoffset, 'classification prediction', fontsize=fontsize)
    plt.text(curr_start + offset, 2 + yoffset, 'regression ground truth', fontsize=fontsize)
    plt.text(curr_start + offset, 3 + yoffset, 'regression prediction', fontsize=fontsize)
    plt.xlim(0, 55000)
    plt.ylim(0, 4)
    suffix = {'f1': '_f1_model.png', 'f0.5': '_f0.5_model.png', 'mIoU': '_mIoU_model.png'}
    plt.title(args.exp_name + f'_{metric}_model')
    plt.savefig(os.path.join('figures', args.exp_name + suffix[metric]))

if __name__ == '__main__':
    # Initialize the model
    print(args.exp_name)
    model = load_model(args.exp_name + '_best_f1.ckpt', num_classes, window_size)
    model.to(device)
    model.eval()
    val(args, model, val_loader, 'f1')
    
    model = load_model(args.exp_name + '_best_f0.5.ckpt', num_classes, window_size)
    model.to(device)
    model.eval()
    val(args, model, val_loader, 'f0.5')
    
    model = load_model(args.exp_name + '_best_miou.ckpt', num_classes, window_size)
    model.to(device)
    model.eval()
    val(args, model, val_loader, 'mIoU')
