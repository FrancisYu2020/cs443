import torch.nn as nn
import torch
import torchvision.models as models
from utils.resnet3d import *

# currently used
class RLSModel(nn.Module):
    def __init__(self, num_classes, window_size=16, pretrained=True):
        super().__init__()
        self.conv = nn.Conv2d(window_size, 3, 3, 3, 1)
        torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out')
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        return self.resnet(x)
    
class RLSRegressionModel(nn.Module):
    def __init__(self, num_classes, window_size=16, pretrained=True):
        super().__init__()
        self.resnet = generate_model(10)
        self.classification_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # initialize conv blocks
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
#         print(x.size())
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        if not self.resnet.no_max_pool:
            x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
#         print(x.size())
        x = self.resnet.layer2(x)
#         print(x.size())
        x = self.resnet.layer3(x)
#         print(x.size())
        x = self.resnet.layer4(x)

#         print(x.size())
        x = self.resnet.avgpool(x)

#         print(x.size())
        x = x.view(x.size(0), -1)

#         print(x.size())
#         exit()
        return self.classification_head(x), self.regression_head(x)