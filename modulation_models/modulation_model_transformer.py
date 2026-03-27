import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of [max_len, d_model] representing position indices
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, input_size=2, d_model=128, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1):
        """
        Transformer architecture based on AMC-Transformer repo.
        Defaults: d_model=128, nhead=8, num_layers=2 match the typical configuration for 128-len signals.
        """
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        
        # 1. Input Embedding (Linear Projection)
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        # batch_first=True is crucial because our input is (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation='gelu' # GELU is often preferred in modern transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, 2)
        
        # 1. Embed and Scale
        # Scaling by sqrt(d_model) is a standard transformer technique to stabilize gradients
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 2. Add Positional Encoding
        x = self.pos_encoder(x)
        
        # 3. Pass through Transformer Encoder
        # Output shape: (Batch, Seq_Len, d_model)
        x = self.transformer_encoder(x)
        
        # 4. Global Average Pooling
        # The repo uses mean pooling over the sequence dimension
        x = x.mean(dim=1) # -> (Batch, d_model)
        
        # 5. Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x