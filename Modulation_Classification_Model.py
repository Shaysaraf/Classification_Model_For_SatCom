# modulation_cnn_lstm_transformer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import upfirdn
import matplotlib.pyplot as plt
import os

# ----------------------------
# PARAMETERS
# ----------------------------
num_symbols = 512
sps = 8
segment_length = 128
num_samples_per_mod = 2000  # reduce to speed up testing
modulations = ['QPSK', '16QAM', '32QAM']
snr_list = [5, 10, 15, 20, 30]  # SNR in dB
batch_size = 128
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = "models_modulation"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# AWGN FUNCTION
# ----------------------------
def add_awgn(iq_signal, snr_db):
    sig_power = np.mean(np.abs(iq_signal)**2)
    snr_linear = 10**(snr_db/10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2)*(np.random.randn(*iq_signal.shape) + 1j*np.random.randn(*iq_signal.shape))
    return iq_signal + noise

# ----------------------------
# DATA GENERATION
# ----------------------------
def generate_symbols(mod, num_symbols):
    if mod == 'QPSK':
        bits = np.random.randint(0, 2, size=(num_symbols*2,))
        mapping = {(0,0):1+1j, (0,1):1-1j, (1,0):-1+1j, (1,1):-1-1j}
        return np.array([mapping[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)])
    else:
        M = int(mod.replace('QAM',''))
        k = int(np.log2(M))
        bits = np.random.randint(0,2,size=(num_symbols*k,))
        # Convert bits to symbols indices (simple method)
        # packbits returns an array of bytes; better to compute indices directly:
        symbols_idx = []
        for i in range(0, len(bits), k):
            idx = 0
            for b in range(k):
                idx = (idx << 1) | int(bits[i+b])
            symbols_idx.append(idx)
        symbols_idx = np.array(symbols_idx)
        m = int(np.sqrt(M))
        I = 2*(symbols_idx % m) - m + 1
        Q = 2*(symbols_idx // m) - m + 1
        return I + 1j*Q

X_list, y_list = [], []
print("Generating modulation dataset...")
for idx, mod in enumerate(modulations):
    for _ in range(num_samples_per_mod):
        symbols = generate_symbols(mod, num_symbols)
        iq = upfirdn([1], symbols, up=sps)  # simple upsample (no pulse shaping)
        snr = np.random.choice(snr_list)
        iq = add_awgn(iq, snr)
        iq /= np.max(np.abs(iq))  # normalize
        iq_arr = np.column_stack((iq.real, iq.imag))
        num_segments = len(iq_arr)//segment_length
        if num_segments == 0:
            continue
        segments = iq_arr[:num_segments*segment_length].reshape(num_segments, segment_length, 2)
        X_list.append(segments)
        y_list.append(np.full((segments.shape[0],), idx))
X = np.vstack(X_list).astype(np.float32)
y = np.hstack(y_list).astype(np.int64)
print("Dataset shape:", X.shape, y.shape)

# ----------------------------
# PyTorch Dataset
# ----------------------------
class IQDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = IQDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ----------------------------
# MODELS
# ----------------------------
# CNN
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*segment_length, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(2, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x = x[:, -1, :]  # last timestep
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Transformer
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*2, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        # x: (B, T, 2)
        x = self.input_proj(x)            # -> (B, T, d_model)
        x = self.transformer(x)           # -> (B, T, d_model)
        x = x.mean(dim=1)                 # pool
        return self.classifier(x)

# ----------------------------
# UTILS
# ----------------------------
def train_model(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total

# ----------------------------
# TRAIN & EVAL FOR EACH MODEL
# ----------------------------
num_classes = len(modulations)
results = {}

# CNN training
cnn = CNNClassifier(num_classes).to(device)
opt = optim.Adam(cnn.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
cnn_train_acc = []
print("\nTraining CNN...")
for ep in range(epochs):
    loss, acc = train_model(cnn, train_loader, opt, crit, device)
    cnn_train_acc.append(acc)
    val_acc = evaluate(cnn, test_loader, device)
    print(f"CNN Epoch {ep+1}/{epochs} loss={loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")
torch.save(cnn.state_dict(), os.path.join(save_dir, "mod_cnn.pth"))
results['cnn'] = evaluate(cnn, test_loader, device)

# LSTM training
lstm = LSTMClassifier(num_classes).to(device)
opt_lstm = optim.Adam(lstm.parameters(), lr=1e-3)
print("\nTraining LSTM...")
for ep in range(epochs):
    loss, acc = train_model(lstm, train_loader, opt_lstm, crit, device)
    val_acc = evaluate(lstm, test_loader, device)
    print(f"LSTM Epoch {ep+1}/{epochs} loss={loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")
torch.save(lstm.state_dict(), os.path.join(save_dir, "mod_lstm.pth"))
results['lstm'] = evaluate(lstm, test_loader, device)

# Transformer training
transformer = TransformerClassifier(num_classes).to(device)
opt_tr = optim.Adam(transformer.parameters(), lr=1e-3)
print("\nTraining Transformer...")
for ep in range(epochs):
    loss, acc = train_model(transformer, train_loader, opt_tr, crit, device)
    val_acc = evaluate(transformer, test_loader, device)
    print(f"Transformer Epoch {ep+1}/{epochs} loss={loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")
torch.save(transformer.state_dict(), os.path.join(save_dir, "mod_transformer.pth"))
results['transformer'] = evaluate(transformer, test_loader, device)

print("\nFinal Test Accuracies (Modulation):", results)

# ----------------------------
# Plot CNN training accuracy
plt.figure(figsize=(8,4))
plt.plot(cnn_train_acc, label='CNN train acc')
plt.title("CNN Training Accuracy (Modulation)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()
plt.show()
