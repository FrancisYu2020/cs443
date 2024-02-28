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

def train(args, model, train_loader, cls_criterion, regression_criterion, optimizer, scheduler):
    '''
    args: input arguments from main function
    model: model to be trained
    train_loader: training data dataloader
    epsilon: the laplace smoothing factor used to prevent division by 0
    cls_criterion: classification loss criterion
    regression_criterion: regression loss criterion
    optimizer: training optimizer
    scheduler: training scheduler 
    '''
    # Train the model
    running_cls_loss, running_regression_loss, total_samples = 0, 0, 0
    running_iou, total_iou_samples = 0, 0
    matrix = np.zeros((2, 2))
    model.train()
    for i, (images, labels, roi_labels) in enumerate(train_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        roi_labels = roi_labels.to(args.device).float()
    
        # Forward pass
        cls_loss, regression_loss = 0, 0
        if args.task == "classification":
            cls_logits = model(images)
            cls_loss += cls_criterion(outputs, labels)
        else:
            cls_logits, regression_logits = model(images)
            cls_loss += cls_criterion(cls_logits, labels)
            regression_loss += regression_criterion(regression_logits, roi_labels, labels)
        loss = args.alpha * cls_loss + (1 - args.alpha) * regression_loss
    
        _, predicted = torch.max(cls_logits, 1)
    
        #compute the iou
        ious = calculate_iou(roi_labels, regression_logits.detach(), print_input=(args.epoch==args.epochs-1))
        running_iou += ious.sum()
        total_iou_samples += ious.size(0)
        #TODO: finish the case when no regression is conducted
    
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
        
    # Calculate metrics
    precision, recall, f1, cls_loss, regression_loss, miou = get_metric_scores(matrix, running_cls_loss, running_regression_loss, running_iou, total_iou_samples)
    wandb.log({"train_cls_loss": running_cls_loss, "train_reg_loss": running_regression_loss, "train_f1": f1, "train_precision": precision, "train_recall": recall, "train_mIoU": running_iou})

        
def val(args, model, val_loader, cls_criterion, regression_criterion):
    '''
    args: input arguments from main function
    model: model to be evaluated 
    train_loader: training data dataloader
    epsilon: the laplace smoothing factor used to prevent division by 0
    cls_criterion: classification loss criterion
    regression_criterion: regression loss criterion
    '''
    matrix = np.zeros((2, 2))
    correct = 0
    model.eval()
    running_cls_loss, running_regression_loss = 0, 0
    running_iou, total_iou_samples = 0, 0
    for i, (images, labels, roi_labels) in enumerate(val_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        roi_labels = roi_labels.to(args.device).float()

        # Forward pass
        cls_loss, regression_loss = 0, 0
        if args.task == 'classification':
            cls_logits = model(images)
            cls_loss += cls_criterion(cls_logits, labels)
        else:
            cls_logits, regression_logits = model(images)
            mask = roi_labels[:, 0] != -1
            cls_loss += cls_criterion(cls_logits, labels)
            regression_loss += regression_criterion(regression_logits, roi_labels, labels)
        loss = cls_loss + regression_loss
        running_cls_loss += cls_loss.item()
        running_regression_loss += regression_loss.item()
        
        _, predicted = torch.max(cls_logits, 1)
        # predicted = torch.ones(outputs.size(0))
        # predicted = torch.from_numpy(np.random.choice([0, 1], size=predicted.size(0)))

        #compute the iou
        ious = calculate_iou(roi_labels, regression_logits, print_input=(args.epoch==args.epochs-1))
        running_iou += ious.sum()
        total_iou_samples += ious.size(0)
        #TODO: finish the case when no regression is conducted
        
        # Calculate the number of correct predictions
        matrix[0][0] += (labels[predicted == 0] == 0).sum()
        matrix[0][1] += (labels[predicted == 1] == 0).sum()
        matrix[1][0] += (labels[predicted == 0] == 1).sum()
        matrix[1][1] += (labels[predicted == 1] == 1).sum()

    # Calculate metrics
    precision, recall, f1, cls_loss, regression_loss, miou = get_metric_scores(matrix, running_cls_loss, running_regression_loss, running_iou, total_iou_samples)
    
    if f1 > args.best_f1:
        # torch.save(model.state_dict(), os.path.join(args.path, 'best_model.ckpt'))
        args.best_f1 = f1
        args.best_epoch = args.epoch + 1
        
    wandb.log({"val_cls_loss": running_cls_loss, "val_reg_loss": running_regression_loss, "val_f1": f1, "val_precision": precision, "val_recall": recall, "val_mIoU": running_iou})
    print(f'Epoch [{args.epoch+1}/{args.epochs}], CLS Loss: {running_cls_loss:.4f}, regression Loss: {running_regression_loss:4f}, f1: {f1:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, mIoU: {running_iou:.2f}%')


def get_metric_scores(confusion_matrix, total_cls_loss, total_regression_loss, total_iou, total_iou_samples, epsilon=0.000001):
    # Calculate accuracy
    total_samples = confusion_matrix.sum()
    
    # decompose confusion matrix
    TN, FP, FN, TP = confusion_matrix.ravel()
    
    # precision, recall, F1
    precision = TP / (TP + FP + epsilon) * 100
    recall = TP / (TP + FN + epsilon) * 100
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    # overall accuracy and class-balanced accuracy (deprecated)
    # overall_accuracy = ((TN + TP) / total_samples) * 100
    # negative_recall = TN / (TN + FP + epsilon) * 100
    # cls_balanced_accuracy = (recall + negative_recall) / 2
    
    # running losses
    cls_loss = total_cls_loss / total_samples
    regression_loss = total_regression_loss / total_samples
    
    # mIoU
    miou = total_iou * 100 / (total_iou_samples + epsilon)
    
    return precision, recall, f1, cls_loss, regression_loss, miou
        