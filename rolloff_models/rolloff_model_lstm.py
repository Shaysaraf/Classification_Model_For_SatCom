import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 2 layers of LSTM to capture complex temporal transitions
        self.lstm1 = nn.LSTM(2, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seg_Len, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Pull the hidden state from the last time step
        x = x[:, -1, :] 
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)