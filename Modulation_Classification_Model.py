import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import upfirdn
import matplotlib.pyplot as plt

# ----------------------------
# PARAMETERS
# ----------------------------
num_symbols = 512
sps = 8
segment_length = 128
num_samples_per_mod = 2000
modulations = ['QPSK', '16QAM', '32QAM']
snr_list = [5, 10, 15, 20, 30]  # SNR in dB
batch_size = 128
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if mod=='QPSK':
        bits = np.random.randint(0,2,size=(num_symbols*2,))
        mapping = {(0,0):1+1j, (0,1):1-1j, (1,0):-1+1j, (1,1):-1-1j}
        return np.array([mapping[(bits[i], bits[i+1])] for i in range(0,len(bits),2)])
    else:
        M = int(mod.replace('QAM',''))
        k = int(np.log2(M))
        bits = np.random.randint(0,2,size=(num_symbols*k,))
        symbols_idx = np.packbits(bits) % M
        m = int(np.sqrt(M))
        I = 2*(symbols_idx % m) - m + 1
        Q = 2*(symbols_idx // m) - m + 1
        return I + 1j*Q

X_list, y_list = [], []

for idx, mod in enumerate(modulations):
    print(f"Generating {mod} sequences...")
    for _ in range(num_samples_per_mod):
        symbols = generate_symbols(mod, num_symbols)
        iq = upfirdn([1], symbols, up=sps)
        # Add noise
        snr = np.random.choice(snr_list)
        iq = add_awgn(iq, snr)
        iq /= np.max(np.abs(iq))
        iq_arr = np.column_stack((iq.real, iq.imag))
        num_segments = len(iq_arr)//segment_length
        segments = iq_arr[:num_segments*segment_length].reshape(num_segments, segment_length, 2)
        X_list.append(segments)
        y_list.append(np.full((segments.shape[0],), idx))

X = np.vstack(X_list).astype(np.float32)
y = np.hstack(y_list).astype(np.int64)
print("Modulation Dataset shape:", X.shape, y.shape)

# ----------------------------
# PyTorch Dataset
# ----------------------------
class IQDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx], self.y[idx]

dataset = IQDataset(X,y)
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset,[train_size,test_size])
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size)

# ----------------------------
# CNN MODEL
# ----------------------------
class CNNClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(2,64,3,padding=1)
        self.conv2 = nn.Conv1d(64,128,3,padding=1)
        self.fc1 = nn.Linear(128*segment_length,128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,num_classes)
    def forward(self,x):
        x = x.permute(0,2,1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

cnn_model = CNNClassifier(len(modulations)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(),lr=0.001)

# ----------------------------
# TRAIN CNN
# ----------------------------
cnn_train_acc = []
for epoch in range(epochs):
    cnn_model.train()
    correct,total=0,0
    for inputs,labels in train_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        _,predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    acc = correct/total
    cnn_train_acc.append(acc)
    print(f"CNN Epoch {epoch+1}/{epochs}, Train Acc: {acc:.4f}")

# ----------------------------
# EVALUATE
# ----------------------------
def evaluate(model,loader):
    model.eval()
    correct,total=0,0
    with torch.no_grad():
        for inputs,labels in loader:
            inputs,labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct/total

cnn_acc = evaluate(cnn_model,test_loader)
print("CNN Test Accuracy (Modulation):", cnn_acc)

# ----------------------------
# Plot CNN training accuracy
plt.figure(figsize=(10,5))
plt.plot(cnn_train_acc,label='CNN train acc')
plt.title("CNN Training Accuracy (Modulation)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()
