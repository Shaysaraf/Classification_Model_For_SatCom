import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input size is 2 (I, Q channels)
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Expects x shape: (Batch, Sequence_Length, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Take the output of the last time step
        x = x[:, -1, :] 
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)