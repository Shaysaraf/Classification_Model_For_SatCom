# rolloff_cnn_lstm_transformer.py
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
num_samples_per_beta = 2000   # reduce to speed up tests if needed
beta_list = [0.1, 0.25, 0.35, 0.5]
snr_list = [5, 10, 15, 20, 30]
batch_size = 128
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = "models_rolloff"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# RRC FILTER (safe for beta=0 edge cases)
# ----------------------------
def rrc_filter(beta, sps, N=101):
    t = np.arange(-N//2, N//2+1)/sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            if beta == 0:
                h[i] = 1.0
            else:
                h[i] = 1 - beta + 4*beta/np.pi
        elif beta != 0 and abs(abs(ti) - 1/(4*beta)) < 1e-12:
            h[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            if beta == 0:
                h[i] = np.sin(np.pi*ti)/(np.pi*ti)
            else:
                h[i] = (np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))) / (np.pi*ti*(1-(4*beta*ti)**2))
    # normalize energy
    return h / np.sqrt(np.sum(h**2) + 1e-12)

# ----------------------------
# AWGN
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
X_list, y_list = [], []
print("Generating roll-off dataset...")
for idx, beta in enumerate(beta_list):
    print("  beta =", beta)
    for _ in range(num_samples_per_beta):
        # QPSK symbols
        bits = np.random.randint(0,2,size=(num_symbols*2,))
        mapping = {(0,0):1+1j, (0,1):1-1j, (1,0):-1+1j, (1,1):-1-1j}
        symbols = np.array([mapping[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)])
        upsampled = upfirdn([1], symbols, up=sps)
        h = rrc_filter(beta, sps, N=101)
        iq = np.convolve(upsampled, h, mode='same')
        snr = np.random.choice(snr_list)
        iq = add_awgn(iq, snr)
        iq /= np.max(np.abs(iq))
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
# MODELS (same architectures as modulation)
# ----------------------------
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

class LSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(2, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.fc1 = nn.Linear(64,64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

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
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ----------------------------
# TRAIN/UTILS
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
# TRAIN & EVAL
# ----------------------------
num_classes = len(beta_list)
results = {}

# CNN
cnn = CNNClassifier(num_classes).to(device)
opt = optim.Adam(cnn.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
cnn_train_acc = []
print("\nTraining CNN (roll-off)...")
for ep in range(epochs):
    loss, acc = train_model(cnn, train_loader, opt, crit, device)
    cnn_train_acc.append(acc)
    val_acc = evaluate(cnn, test_loader, device)
    print(f"CNN Epoch {ep+1}/{epochs} loss={loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")
torch.save(cnn.state_dict(), os.path.join(save_dir, "rolloff_cnn.pth"))
results['cnn'] = evaluate(cnn, test_loader, device)

# LSTM
lstm = LSTMClassifier(num_classes).to(device)
opt_lstm = optim.Adam(lstm.parameters(), lr=1e-3)
print("\nTraining LSTM (roll-off)...")
for ep in range(epochs):
    loss, acc = train_model(lstm, train_loader, opt_lstm, crit, device)
    val_acc = evaluate(lstm, test_loader, device)
    print(f"LSTM Epoch {ep+1}/{epochs} loss={loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")
torch.save(lstm.state_dict(), os.path.join(save_dir, "rolloff_lstm.pth"))
results['lstm'] = evaluate(lstm, test_loader, device)

# Transformer
transformer = TransformerClassifier(num_classes).to(device)
opt_tr = optim.Adam(transformer.parameters(), lr=1e-3)
print("\nTraining Transformer (roll-off)...")
for ep in range(epochs):
    loss, acc = train_model(transformer, train_loader, opt_tr, crit, device)
    val_acc = evaluate(transformer, test_loader, device)
    print(f"Transformer Epoch {ep+1}/{epochs} loss={loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")
torch.save(transformer.state_dict(), os.path.join(save_dir, "rolloff_transformer.pth"))
results['transformer'] = evaluate(transformer, test_loader, device)

print("\nFinal Test Accuracies (Roll-off):", results)

# ----------------------------
# Plot CNN training accuracy
plt.figure(figsize=(8,4))
plt.plot(cnn_train_acc, label='CNN train acc')
plt.title("CNN Training Accuracy (Roll-off)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()
plt.show()
