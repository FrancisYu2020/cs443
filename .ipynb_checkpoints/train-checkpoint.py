import torch
import torch.nn as nn
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
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

data_paths = ['12-13-2023', '02-15-2024', '02-17-2024']
if args.cross_val_type == 0:
    train_data = np.concatenate([np.load(os.path.join(path, args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32) for path in data_paths], axis=0)
    train_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_label_train.npy')) for path in data_paths], axis=0)
    val_data = np.concatenate([np.load(os.path.join(path, args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32) for path in data_paths], axis=0)
    val_label = np.concatenate([np.load(os.path.join(path, args.window_size + 'EMG_label_val.npy')) for path in data_paths], axis=0)
else:
    leave_out_idx = args.cross_val_type - 1
    train_data = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
    train_label = np.concatenate([np.load(os.path.join(data_paths[i], args.window_size + 'EMG_label_train.npy')) for i in range(len(data_paths)) if i != leave_out_idx], axis=0)
    val_data = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'sensing_mat_data_train.npy')).astype(np.float32), np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'sensing_mat_data_val.npy')).astype(np.float32)], axis=0)
    val_label = np.concatenate([np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_label_train.npy')), np.load(os.path.join(data_paths[leave_out_idx], args.window_size + 'EMG_label_val.npy'))], axis=0)

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
    transforms.RandomRotation(degrees=(-180, 180)),
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
train_dataset = RLSDataset(train_data, train_label, transform=transform)
val_dataset = RLSDataset(val_data, val_label, transform=val_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RLSModel(num_classes, window_size=window_size, pretrained=False)
model.to(device)

# Loss and optimizer
img_num_per_cls = np.unique(train_label, return_counts=True)[1]
# print(img_num_per_cls)
def CB_lossFunc(logits, labelList): #defince CB loss function
    return CB_loss(labelList, logits, img_num_per_cls, num_classes, "softmax", 0.9999, 2.0, device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

best_f1 = -np.inf
best_epoch = None
df = pd.DataFrame(columns=["epoch", "validation_loss", "overall_accuracy", "cls_balanced_accuracy", "f1", "precision", "recall"])
training_losses, val_losses, training_accs, val_accs = [], [], [], []
epsilon = 0.000001  # laplace smoothing

# Train the model
feature_learning_epochs = num_epochs * 1
for epoch in range(num_epochs):
#     print(model.fn.weight)
    running_loss, total_samples = 0, 0
    correct = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        total_samples += images.size(0)
#         break
#         if epoch <= 3:
#             break
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        if epoch < feature_learning_epochs:
            loss = criterion(outputs, labels)
        else:
            loss = CB_lossFunc(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)

        # Calculate the number of correct predictions
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    running_loss /= total_samples
    # Calculate accuracy
    accuracy = (correct / total_samples) * 100
    training_losses.append(running_loss)
    training_accs.append(accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Training accuracy: {accuracy:.2f}%')
    
    matrix = np.zeros((2, 2))
    correct = 0
    model.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
#         predicted = torch.ones(outputs.size(0))
#         predicted = torch.from_numpy(np.random.choice([0, 1], size=predicted.size(0)))
#         print(predicted)

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
    running_loss /= total_samples
    val_losses.append(running_loss)
    val_accs.append(cls_balanced_accuracy)
    if f1 > best_f1:
        torch.save(model.state_dict(), os.path.join(args.path, 'best_model.ckpt'))
        best_f1 = f1
        best_epoch = epoch + 1

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, overall: {overall_accuracy:.2f}%, cls_balanced: {cls_balanced_accuracy:.2f}, f1: {f1:.2f}, precision: {precision:.2f}, recall: {recall:.2f}')
#     continue
    
    df.loc[len(df)] = [epoch + 1, running_loss, overall_accuracy, cls_balanced_accuracy, f1, precision, recall]
    log_filename = args.window_size + 'log.csv' if args.exp_label is None else f'{args.window_size}{args.exp_label}_log.csv'
    df.to_csv(os.path.join(args.path, log_filename))

df.overall_accuracy.plot()
df.cls_balanced_accuracy.plot()
df.f1.plot()
df.precision.plot()
df.recall.plot()
plt.ylim(0, 100)
plt.legend()
plt.title('Validation metrics vs. epochs')
plt.savefig(os.path.join(args.path, args.window_size + 'metrics.png'))
# plt.plot(1 + np.arange(num_epochs), training_losses, label='training loss')
# plt.plot(1 + np.arange(num_epochs), val_losses, label='validation loss')
# plt.legend()
# plt.title('Training and validation loss')
# plt.savefig(os.path.join(args.path, 'SGD_0.01_debug_loss.png'))
# plt.clf()
# plt.plot(1 + np.arange(num_epochs), training_accs, label='training loss')
# plt.plot(1 + np.arange(num_epochs), val_accs, label='validation loss')
# plt.ylim(0, 100)
# plt.legend()
# plt.title('Training and validation accuracy')
# plt.savefig(os.path.join(args.path, 'SGD_0.01_debug_accuracy.png'))
print(f'The best validation accuracy is {best_f1:.2f} at epoch {best_epoch}')