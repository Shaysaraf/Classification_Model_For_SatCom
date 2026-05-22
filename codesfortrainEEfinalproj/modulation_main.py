import os
import json
import time
import copy
import gc
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- IMPORTS ---
# Ensure these files are in your working directory or python path
from modulation_models.modulation_model_resnet34 import CNNClassifier
from data_manager import SatComDataManager 

# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class TrainConfig:
    segment_length: int = 512       # Increased from 128 for better feature resolution
    modulations: tuple = ('16apsk', '8psk', 'qpsk') # Lowercase to match database mapping
    batch_size: int = 64            # Reduced slightly due to longer segment length/memory
    epochs: int = 400
    early_stop_patience: int = 100    # Stops if no improvement for 70 epochs
    learning_rate: float = 1e-3 
    weight_decay: float = 1e-3      # Stronger decay for deeper ResNet
    label_smoothing: float = 0.05
    seed: int = 42
    save_dir: str = "models_results"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def write_log(message, log_path):
    with open(log_path, "a") as f:
        f.write(message + "\n")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================================================================
# DATASET (Updated for 4-Channel I/Q/A/P)
# ==============================================================================
class LazyIQDataset(Dataset):
    def __init__(self, data_mgr, modulations, segment_length, mode='train'):
        self.data_mgr = data_mgr
        self.segment_length = segment_length
        self.mode = mode 
        self.file_index = []
        
        if not data_mgr.master_json_path.exists():
            raise FileNotFoundError(f"Metadata not found at {data_mgr.master_json_path}")

        print(f"Loading metadata for {mode} mode...")
        with open(data_mgr.master_json_path, 'r') as f:
            database = json.load(f)

        files = sorted(data_mgr.get_sample_list())
        mod_map = {m.lower().strip(): i for i, m in enumerate(modulations)}
        
        for file_path in files:
            candidate = file_path.stem
            found_entry = None
            
            while candidate:
                if candidate in database:
                    found_entry = database[candidate]
                    break
                if '_' in candidate: 
                    candidate = candidate.rsplit('_', 1)[0]
                else: 
                    break
            
            if found_entry:
                raw_mod = str(found_entry.get("modcod", "")).strip()
                if raw_mod:
                    current_mod = raw_mod.split()[0].lower()
                    if current_mod in mod_map:
                        self.file_index.append((file_path, mod_map[current_mod]))

        print(f"Dataset ready. Found {len(self.file_index)} samples.")

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        file_path, label = self.file_index[idx]
        iq_data = self.data_mgr.load_iq_sample(file_path)
        
        if iq_data is None:
            # Return zero tensor with 4 channels if file fails to load
            return torch.zeros((4, self.segment_length)), torch.tensor(label)

        # 1. Normalize
        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0: 
            iq_data /= (max_val + 1e-6)
            
        # 2. Augmentation (Random Phase Rotation)
        if self.mode == 'train':
            theta = np.random.uniform(0, 2 * np.pi)
            iq_data = iq_data * np.exp(1j * theta)

        # 3. Feature Engineering: Extract Amplitude and Phase
        # This helps the CNN distinguish between 8PSK (constant amp) and 16APSK (varying amp)
        amp = np.abs(iq_data)
        phase = np.angle(iq_data)
        
        # Stack into 4 channels: [I, Q, Amp, Phase]
        iq_arr = np.column_stack((iq_data.real, iq_data.imag, amp, phase))
        
        # 4. Pad/Crop
        if len(iq_arr) >= self.segment_length:
            if self.mode == 'train':
                start = np.random.randint(0, len(iq_arr) - self.segment_length + 1)
            else:
                start = (len(iq_arr) - self.segment_length) // 2
            segment = iq_arr[start:start + self.segment_length]
        else:
            padding = np.zeros((self.segment_length - len(iq_arr), 4), dtype=iq_arr.dtype)
            segment = np.vstack((iq_arr, padding))
            
        # Transpose to (Channels, Length) -> (4, 512)
        segment = segment.transpose() 
            
        return torch.from_numpy(segment).float(), torch.tensor(label, dtype=torch.long)

# ==============================================================================
# TRAINING ENGINE (Updated with Early Stopping)
# ==============================================================================
class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.epochs, 
            eta_min=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        self.log_file = os.path.join(self.config.save_dir, "training_log.txt")
        
        # History to track both metrics
        self.history = {
            'train_loss': [], 
            'train_acc': [], 
            'val_acc': []
        }

    def train_epoch(self):
        self.model.train()
        running_loss, t_correct, t_total = 0.0, 0, 0
        
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            t_total += labels.size(0)
            t_correct += (preds == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        return running_loss / t_total, t_correct / t_total

    def validate_epoch(self):
        self.model.eval()
        v_correct, v_total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
                
        return v_correct / v_total if v_total > 0 else 0

    def fit(self, name="ResNet_CNN"):
        if os.path.exists(self.log_file): os.remove(self.log_file)
        write_log(f"--- Starting {name} Training ---", self.log_file)
        
        best_acc = 0.0
        epochs_no_improve = 0
        best_wts = copy.deepcopy(self.model.state_dict())
        
        print(f"{'Epoch':<8} | {'Loss':<8} | {'Tr Acc':<8} | {'Val Acc':<8} | {'Best':<8} | {'Time':<5}")
        print("-" * 60)

        for ep in range(self.config.epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_acc = self.validate_epoch()
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Improvement Check
            status = " "
            if val_acc > best_acc:
                best_acc = val_acc
                best_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                status = "*" 
            else:
                epochs_no_improve += 1
                
            elapsed = time.time() - start_time
            
            # Formatted Console Output
            msg = (f"{ep+1:03d}/{self.config.epochs:<3} | {train_loss:.4f}   | "
                   f"{train_acc:.2%}   | {val_acc:.2%} {status} | {best_acc:.2%}  | {elapsed:.1f}s")
            print(msg)
            write_log(msg, self.log_file)

            if epochs_no_improve >= self.config.early_stop_patience:
                print(f"\n[!] Early stopping: No improvement for {self.config.early_stop_patience} epochs.")
                break

        self.model.load_state_dict(best_wts)
        torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f"best_{name.lower()}.pth"))
        self._plot_results(name)
        return best_acc

    def _plot_results(self, name):
        plt.figure(figsize=(12, 5))
        
        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], label='Train Accuracy', color='blue', alpha=0.7)
        plt.plot(self.history['val_acc'], label='Val Accuracy', color='green', linewidth=2)
        plt.title(f'{name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Train Loss', color='red')
        plt.title(f'{name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, f"{name}_metrics.png"))
        plt.close()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    config = TrainConfig()
    os.makedirs(config.save_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Initialize Manager
    data_mgr = SatComDataManager()
    
    # Initialize Datasets (Mapping to same source but different augmentation modes)
    train_dataset = LazyIQDataset(data_mgr, config.modulations, config.segment_length, mode='train')
    eval_dataset = LazyIQDataset(data_mgr, config.modulations, config.segment_length, mode='eval')

    if len(train_dataset) == 0:
        print("Error: No data found for the specified modulations.")
        return

    # Train/Val Split (80/20)
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_ds = Subset(train_dataset, indices[:train_size])
    val_ds = Subset(eval_dataset, indices[train_size:])
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model (Accepts 4 channels now)
    num_classes = len(config.modulations)
    model = CNNClassifier(num_classes=num_classes, segment_length=config.segment_length)
    
    print(f"\n>>> Training {type(model).__name__} on {config.device}")
    print(f">>> Samples: {len(train_ds)} train, {len(val_ds)} val")
    print(f">>> Input Shape: (4, {config.segment_length})")
    
    trainer = Trainer(model, config, train_loader, val_loader)
    best_accuracy = trainer.fit("ResNet_CNN")
    
    cleanup_memory()
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"=== BEST VALIDATION ACCURACY: {best_accuracy*100:.2f}% ===")

if __name__ == "__main__":
    main()