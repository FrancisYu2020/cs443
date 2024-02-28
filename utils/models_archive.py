import torch.nn as nn
import torch
import torchvision.models as models

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
#         print(out.size(), identity.size())
        out += identity
        out = self.relu(out)

        return out

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
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(window_size, window_size, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(window_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, window_size, 2)
        self.layer2 = self._make_layer(BasicBlock, window_size, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, window_size, 2, stride=2)
#         self.layer4 = self._make_layer(BasicBlock, window_size, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(window_size, num_classes)
        
#         self.conv_adapt = nn.Conv2d(window_size, 3, 3, 1)
#         self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
# #         torch.nn.init.kaiming_normal_(self.conv0.weight, a=0, mode='fan_out')
# #         torch.nn.init.kaiming_normal_(self.conv_adapt.weight, a=0, mode='fan_out')
#         self.resnet = models.resnet18(pretrained=pretrained)
        self.classification_head = nn.Sequential(
            nn.Linear(window_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.regression_head = nn.Sequential(
#             nn.Linear(self.resnet.fc.in_features, 2)
            nn.Linear(window_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # initialize conv blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
#         x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
#         print(x.max(), x.min())
        return self.classification_head(x), self.regression_head(x)
#         return self.classification_head(x), torch.clamp(self.regression_head(x), -2, 2)
       
#     def forward(self, x):
#         x = self.conv0(x)
#         regression_x = self.global_average_pool(x).view(x.size(0), -1)
#         x = self.conv_adapt(x)
# #         print(regression_x.size())
        
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)

#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)

#         x = self.resnet.avgpool(x)
#         x = torch.flatten(x, 1)
# #         print(x.max(), x.min())
#         return self.classification_head(x), torch.clamp(self.regression_head(regression_x), -2, 2)
    
    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(planes * block.expansion, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation
            )
        )
#         self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)
    
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