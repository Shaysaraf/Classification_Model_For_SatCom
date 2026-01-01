import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        # Project the 2 (I, Q) inputs to the model dimension
        self.input_proj = nn.Linear(2, d_model)
        
        # Standard Transformer Encoder architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seg_Len, 2)
        x = self.input_proj(x)
        
        # Transformer treats the sequence as a whole
        x = self.transformer(x)
        
        # Global Average Pooling across the time (segment) dimension
        x = x.mean(dim=1) 
        
        return self.fc(x)