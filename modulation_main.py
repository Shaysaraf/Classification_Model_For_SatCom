import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
import warnings
import sys
import time
import copy
import gc
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch.multiprocessing

# --- JETSON FIX ---
torch.multiprocessing.set_sharing_strategy('file_system')

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

# --- IMPORTS ---
# Ensure these files are in the 'modulation_models' directory
from modulation_models.modulation_model_cnn import CNNClassifier
from modulation_models.modulation_model_lstm import LSTMClassifier
from modulation_models.modulation_model_transformer import TransformerClassifier
from data_manager import SatComDataManager 

# ==============================================================================
# HELPER: LOGGING
# ==============================================================================
def write_log(message, log_path):
    print(message)
    with open(log_path, "a") as f:
        f.write(message + "\n")

# ==============================================================================
# DATASET
# ==============================================================================
class LazyIQDataset(Dataset):
    def __init__(self, data_mgr, modulations, segment_length, mode='train'):
        self.data_mgr = data_mgr
        self.segment_length = segment_length
        self.mode = mode 
        self.file_index = []
        
        if not data_mgr.master_json_path.exists():
            raise FileNotFoundError(f"Metadata not found at {data_mgr.master_json_path}")

        print(f"Loading metadata...")
        with open(data_mgr.master_json_path, 'r') as f:
            database = json.load(f)

        files = sorted(data_mgr.get_sample_list())
        mod_map = {m.lower().strip(): i for i, m in enumerate(modulations)}
        
        print(f"Scanning {len(files)} files...")
        for file_path in files:
            name_stem = file_path.stem
            candidate = name_stem
            found_entry = None
            
            # Simple metadata lookup
            while candidate:
                if candidate in database:
                    found_entry = database[candidate]
                    break
                if '_' in candidate: candidate = candidate.rsplit('_', 1)[0]
                else: break
            
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
            return torch.zeros((2, self.segment_length)), torch.tensor(label)

        # Normalize
        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0: iq_data /= (max_val + 1e-6)
            
        # Augmentation (Train only) - Random Phase Rotation
        if self.mode == 'train':
            theta = np.random.uniform(0, 2 * np.pi)
            iq_data = iq_data * np.exp(1j * theta)

        iq_arr = np.column_stack((iq_data.real, iq_data.imag))
        
        # Pad/Crop
        if len(iq_arr) >= self.segment_length:
            if self.mode == 'train':
                start = np.random.randint(0, len(iq_arr) - self.segment_length + 1)
            else:
                start = (len(iq_arr) - self.segment_length) // 2
            segment = iq_arr[start:start + self.segment_length]
        else:
            padding = np.zeros((self.segment_length - len(iq_arr), 2), dtype=iq_arr.dtype)
            segment = np.vstack((iq_arr, padding))
            
        # Shape: (128, 2). PyTorch prefers (2, 128) for CNNs, but (128, 2) for RNNs.
        # We will standardize to (2, 128) here and permute inside the loop for RNNs.
        segment = segment.transpose() 
            
        return torch.from_numpy(segment).float(), torch.tensor(label, dtype=torch.long)

# ==============================================================================
# TRAIN ENGINE
# ==============================================================================
def train_model(model, name, train_loader, test_loader, params):
    log_file = os.path.join(params['save_dir'], f"{name}_log.txt")
    if os.path.exists(log_file): os.remove(log_file)
    
    device = params['device']
    model.to(device)
    
    # --- MODEL SPECIFIC OPTIMIZERS ---
    if name == "Transformer":
        # Transformers need Weight Decay and lower LR
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        use_early_stopping = False
        patience = 0
        min_epochs_before_early_stop = 0
    elif name == "LSTM":
        # LSTMs can handle slightly higher LR
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        use_early_stopping = True
        patience = params.get('lstm_patience', 7)
        min_epochs_before_early_stop = params.get('lstm_min_epochs', 0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    else: # CNN
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        use_early_stopping = True
        patience = params.get('cnn_patience', 10)
        min_epochs_before_early_stop = params.get('cnn_min_epochs', 20)

    model_epochs = params.get(f"{name.lower()}_epochs", params['epochs'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    if name == "CNN":
        criterion = nn.CrossEntropyLoss(label_smoothing=params.get('cnn_label_smoothing', 0.05))
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    write_log(f"--- Starting {name} | Device: {device} ---", log_file)

    for ep in range(model_epochs):
        start = time.time()
        model.train()
        t_correct, t_total = 0, 0
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # RESHAPE FIX:
            # CNN: (Batch, 2, 128) -> OK
            # LSTM/Trans: (Batch, 128, 2) -> Permute
            if name in ["LSTM", "Transformer"]:
                inputs = inputs.permute(0, 2, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping (Helps LSTM/Transformer stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            t_total += labels.size(0)
            t_correct += (preds == labels).sum().item()
        
        epoch_loss = running_loss / t_total
        epoch_acc = t_correct / t_total
        
        # --- VALIDATION ---
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if name in ["LSTM", "Transformer"]:
                    inputs = inputs.permute(0, 2, 1)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        
        val_acc = v_correct / v_total if v_total > 0 else 0
        
        # Logging
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        # Save Best & Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0 # Reset
        else:
            early_stop_counter += 1
            
        elapsed = time.time() - start
        msg = f"Ep {ep+1}/{model_epochs} | Loss: {epoch_loss:.4f} | Tr: {epoch_acc:.2%} | Val: {val_acc:.2%} | Best: {best_acc:.2%}"
        write_log(msg, log_file)
        
        if use_early_stopping and (ep + 1) >= min_epochs_before_early_stop and early_stop_counter >= patience:
            write_log(f"Early stopping triggered at epoch {ep+1}", log_file)
            break

    # Load best weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(params['save_dir'], f"best_{name.lower()}.pth"))
    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title(f'{name} Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Loss', color='red')
    plt.title(f'{name} Loss')
    plt.savefig(os.path.join(params['save_dir'], f"{name}_results.png"))
    plt.close()

    if name == "Transformer":
        model_config = {
            'num_classes': model.fc.out_features,
            'input_size': model.input_projection.in_features,
            'd_model': model.d_model,
            'nhead': model.transformer_encoder.layers[0].self_attn.num_heads,
            'num_layers': len(model.transformer_encoder.layers),
            'dim_feedforward': model.transformer_encoder.layers[0].linear1.out_features,
            'dropout': model.dropout.p,
        }

        checkpoint = {
            'model_name': name,
            'best_val_acc': best_acc,
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'train_history': history,
            'training_params': make_json_serializable(params),
            'num_trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        torch.save(checkpoint, os.path.join(params['save_dir'], "best_transformer_checkpoint.pth"))

        with open(os.path.join(params['save_dir'], "transformer_params.json"), "w") as f:
            json.dump({
                'model_config': model_config,
                'training_params': make_json_serializable(params),
                'best_val_acc': best_acc,
                'num_trainable_params': checkpoint['num_trainable_params'],
            }, f, indent=2)

        with open(os.path.join(params['save_dir'], "Transformer_history.json"), "w") as f:
            json.dump(make_json_serializable(history), f, indent=2)

        plt.figure(figsize=(6, 4))
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Val')
        plt.title('Transformer Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(params['save_dir'], "Transformer_accuracy.png"))
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(history['train_loss'], label='Loss', color='red')
        plt.title('Transformer Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(params['save_dir'], "Transformer_loss.png"))
        plt.close()
    
    return best_acc


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--transformer', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    params = {
        'segment_length': 128,
        'modulations': ['16APSK', '8PSK', 'QPSK'], 
        'batch_size': 32, # 32 is safer for Jetson RAM
        'epochs': 50,
        'cnn_epochs': 50,
        'transformer_epochs': 50,
        'lstm_epochs': 50,
        'seed': 42,
        'cnn_patience': 8,
        'cnn_min_epochs': 20,
        'cnn_label_smoothing': 0.05,
        'transformer_patience': 12,
        'transformer_min_epochs': 15,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': "models_results_v2"
    }
    
    os.makedirs(params['save_dir'], exist_ok=True)
    
    if not (args.cnn or args.lstm or args.transformer or args.all):
        print("Usage: python main.py --cnn --lstm --transformer --all")
        return

    data_mgr = SatComDataManager()
    train_dataset = LazyIQDataset(data_mgr, params['modulations'], params['segment_length'], mode='train')
    eval_dataset = LazyIQDataset(data_mgr, params['modulations'], params['segment_length'], mode='eval')

    if len(train_dataset) == 0:
        print("No data found.")
        return

    if len(train_dataset) != len(eval_dataset):
        print("Dataset mismatch between train/eval views.")
        return

    # Deterministic split with shared indices across train/eval views
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    generator = torch.Generator().manual_seed(params['seed'])
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    train_ds = Subset(train_dataset, train_idx)
    test_ds = Subset(eval_dataset, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=False, num_workers=0)

    num_classes = len(params['modulations'])
    results = {}

    if args.cnn or args.all:
        print("\n>>> Training CNN...")
        acc = train_model(CNNClassifier(num_classes), "CNN", train_loader, test_loader, params)
        results["CNN"] = acc
        cleanup_memory()

    if args.lstm or args.all:
        print("\n>>> Training LSTM...")
        # LSTM input size is 2 (I, Q)
        acc = train_model(LSTMClassifier(num_classes, input_size=2), "LSTM", train_loader, test_loader, params)
        results["LSTM"] = acc
        cleanup_memory()

    if args.transformer or args.all:
        print("\n>>> Training Transformer...")
        acc = train_model(TransformerClassifier(num_classes), "Transformer", train_loader, test_loader, params)
        results["Transformer"] = acc
        cleanup_memory()

    print("\n=== FINAL TEST ACCURACY ===")
    for k, v in results.items():
        print(f"{k}: {v*100:.2f}%")

if __name__ == "__main__":
    main()