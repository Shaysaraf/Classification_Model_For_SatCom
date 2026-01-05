import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # If the input shape doesn't match the output (due to stride or channel change),
        # we must downsample the identity (shortcut) to match.
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class CNNClassifier(nn.Module):
    def __init__(self, num_classes, segment_length=128):
        super(CNNClassifier, self).__init__()
        
        # Standard ResNet configurations often use [2, 2, 2, 2] blocks
        layers = [2, 2, 2, 2] 
        
        self.inplanes = 64
        
        # --- Initial Stem ---
        # Input: (Batch, 2, 128)
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # --- Residual Layers ---
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # --- Classification Head ---
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        """
        Builds a ResNet layer composed of 'blocks' number of ResidualBlocks.
        Only the first block handles the stride/downsampling.
        """
        downsample = None
        
        # If stride != 1 or channels change, we need a 1x1 conv on the shortcut
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        # First block (potentially strided)
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        # Subsequent blocks (always stride=1)
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, 2, 128)
        
        x = self.conv1(x)       # -> (64, 64)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # -> (64, 32)

        x = self.layer1(x)      # -> (64, 32)
        x = self.layer2(x)      # -> (128, 16)
        x = self.layer3(x)      # -> (256, 8)
        x = self.layer4(x)      # -> (512, 4)

        x = self.avgpool(x)     # -> (512, 1)
        x = torch.flatten(x, 1) # -> (512)
        x = self.fc(x)          # -> (num_classes)
        
        return x