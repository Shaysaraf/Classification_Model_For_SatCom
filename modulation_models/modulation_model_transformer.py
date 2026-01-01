import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global Average Pooling
        return self.fc(x)