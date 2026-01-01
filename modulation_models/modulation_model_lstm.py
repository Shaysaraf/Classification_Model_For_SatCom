import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(2, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take the last time step
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)