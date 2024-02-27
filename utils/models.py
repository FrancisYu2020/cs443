import torch.nn as nn
import torch
import torchvision.models as models

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
        self.conv = nn.Conv2d(window_size, 3, 3, 3, 1)
        torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out')
        self.resnet = models.resnet18(pretrained=pretrained)
        self.classification_head = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classification_head(x), self.regression_head(x)
    
# ViT based
# class RLSModel(nn.Module):
#     def __init__(self, num_classes, window_size=16, pretrained=True):
#         super().__init__()
# #         self.fn = nn.Linear(window_size * 256, 2)
#         self.fn = nn.Sequential(
#             nn.Linear(window_size * 256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2)
#         )
# #         self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=False)
# #         self.conv = nn.Conv2d(window_size, 3, 3, 1, 1)
# #         torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out')
# #         self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
# #         for param in self.vit.parameters():
# #             param.requires_grad = False
# #         self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
# #         torch.nn.init.kaiming_normal_(self.vit.head.weight, a=0, mode='fan_out')
    
#     def forward(self, x):
# #         return self.vit(self.conv(x))
#         return self.fn(x.flatten(start_dim=1))
#         x = self.conv(self.upsample(x))
#         return self.vit(x)

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel=3, stride=2, padding=1):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
#             nn.ReLU(),
#             nn.BatchNorm2d(out_channels)
#         )
    
#     def forward(self, x):
#         return self.model(x)
    
# class RLSModel(nn.Module):
#     def __init__(self, num_classes, window_size=16, pretrained=True):
#         super().__init__()
# #         self.conv1 = ConvBlock(window_size, 8)
# #         self.conv2 = ConvBlock(8, 4)
#         self.fc = nn.Linear(256 * window_size, 2)
        
#         for module in self.modules():
#             try:
#                 torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
#             except:
#                 pass
    
#     def forward(self, x):
# #         x = self.conv1(x)
# #         x = self.conv2(x).flatten(start_dim=1)
# #         x = self.conv3(x).flatten(dim=1)
#         return self.fc(x.flatten(start_dim=1))