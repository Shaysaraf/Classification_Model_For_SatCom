import torch
import torch.nn as nn

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock2D, self).__init__()
        # Use (1, 3) kernels to mimic 1D convolutions on a 2D DPU hardware engine
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return self.dropout(out)

class ResNet34XilinxClassifier(nn.Module):
    def __init__(self, num_classes, segment_length=512):
        super(ResNet34XilinxClassifier, self).__init__()
        # ResNet-34 block distribution profile
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        
        # Input channels = 4 (I, Q, Amp, Phase) converted to 2D space
        self.conv1 = nn.Conv2d(4, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=(1, 1), stride=(1, stride), bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock2D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock2D(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Automatically insert the dummy height dimension: (Batch, 4, 512) -> (Batch, 4, 1, 512)
        if x.dim() == 3:
            x = x.unsqueeze(2)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)