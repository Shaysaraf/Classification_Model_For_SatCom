import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, num_classes, segment_length):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * segment_length, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)