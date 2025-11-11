import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --------------------------------------------------------
# 1. Dataset Class
# --------------------------------------------------------
class DVBS1Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------------------------------------------
# 2. CNN Model Definition
# --------------------------------------------------------
class SimpleIQCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleIQCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),
            nn.Linear(64 * (2048 // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------
# 3. Main Training Loop
# --------------------------------------------------------
def main():
    # Load data
    X = np.load("data_processed/X_qpsk1_4.npy")
    y = np.load("data_processed/y_qpsk1_4.npy")

    # Example: if you only have one modulation, create dummy labels
    # (for testing multi-class setup, you’ll need more than one label)
    num_classes = len(np.unique(y))
    print(f"[INFO] Dataset: {X.shape}, Classes: {num_classes}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if num_classes > 1 else None
    )

    train_ds = DVBS1Dataset(X_train, y_train)
    test_ds = DVBS1Dataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Model, loss, optimizer
    model = SimpleIQCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] Using device: {device}")

    # Train
    for epoch in range(10):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/10]  Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total if total > 0 else 0
    print(f"[RESULT] Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
