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
from modulation_model_resnet import ResNet34XilinxClassifier
from data_manager import SatComDataManager 

@dataclass
class TrainConfig:
    segment_length: int = 512       
    modulations: tuple = ('16apsk', '8psk', 'qpsk') 
    batch_size: int = 64            
    epochs: int = 400
    early_stop_patience: int = 100    
    learning_rate: float = 1e-3 
    weight_decay: float = 1e-3      
    label_smoothing: float = 0.05
    seed: int = 42
    save_dir: str = "models_results_resnet34"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def write_log(message, log_path):
    with open(log_path, "a") as f:
        f.write(message + "\n")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
            return torch.zeros((4, self.segment_length)), torch.tensor(label)

        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0: 
            iq_data /= (max_val + 1e-6)
            
        if self.mode == 'train':
            theta = np.random.uniform(0, 2 * np.pi)
            iq_data = iq_data * np.exp(1j * theta)

        amp = np.abs(iq_data)
        phase = np.angle(iq_data)
        iq_arr = np.column_stack((iq_data.real, iq_data.imag, amp, phase))
        
        if len(iq_arr) >= self.segment_length:
            if self.mode == 'train':
                start = np.random.randint(0, len(iq_arr) - self.segment_length + 1)
            else:
                start = (len(iq_arr) - self.segment_length) // 2
            segment = iq_arr[start:start + self.segment_length]
        else:
            padding = np.zeros((self.segment_length - len(iq_arr), 4), dtype=iq_arr.dtype)
            segment = np.vstack((iq_arr, padding))
            
        segment = segment.transpose() 
        return torch.from_numpy(segment).float(), torch.tensor(label, dtype=torch.long)

class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=1e-6)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        self.log_file = os.path.join(self.config.save_dir, "training_log.txt")
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

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

    def fit(self, name="ResNet34_DPU"):
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
            
            status = " "
            if val_acc > best_acc:
                best_acc = val_acc
                best_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                status = "*" 
            else:
                epochs_no_improve += 1
                
            elapsed = time.time() - start_time
            msg = f"{ep+1:03d}/{self.config.epochs:<3} | {train_loss:.4f}   | {train_acc:.2%}   | {val_acc:.2%} {status} | {best_acc:.2%}  | {elapsed:.1f}s"
            print(msg)
            write_log(msg, self.log_file)

            if epochs_no_improve >= self.config.early_stop_patience:
                print(f"\n[!] Early stopping: No improvement for {self.config.early_stop_patience} epochs.")
                break

        self.model.load_state_dict(best_wts)
        torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f"best_{name.lower()}.pth"))
        return best_acc

def main():
    config = TrainConfig()
    os.makedirs(config.save_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data_mgr = SatComDataManager()
    
    train_dataset = LazyIQDataset(data_mgr, config.modulations, config.segment_length, mode='train')
    eval_dataset = LazyIQDataset(data_mgr, config.modulations, config.segment_length, mode='eval')

    if len(train_dataset) == 0:
        print("Error: No data found for the specified modulations.")
        return

    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_ds = Subset(train_dataset, indices[:train_size])
    val_ds = Subset(eval_dataset, indices[train_size:])
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(config.modulations)
    model = ResNet34XilinxClassifier(num_classes=num_classes, segment_length=config.segment_length)
    
    print(f"\n>>> Training {type(model).__name__} on {config.device}")
    print(f">>> Samples: {len(train_ds)} train, {len(val_ds)} val")
    print(f">>> Input Shape: (4, {config.segment_length})")
    
    trainer = Trainer(model, config, train_loader, val_loader)
    best_accuracy = trainer.fit("ResNet34_DPU")
    
    cleanup_memory()
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"=== BEST VALIDATION ACCURACY: {best_accuracy*100:.2f}% ===")

if __name__ == "__main__":
    main()