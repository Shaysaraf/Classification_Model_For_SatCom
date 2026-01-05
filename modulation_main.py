import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
import json
import numpy as np
import warnings
import sys
import time  # Added for timing
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

# --- MATPLOTLIB SETUP ---
# Must be done before importing pyplot
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- IMPORTS ---
try:
    from modulation_models.modulation_model_cnn import CNNClassifier
    from modulation_models.modulation_model_lstm import LSTMClassifier
    from modulation_models.modulation_model_transformer import TransformerClassifier
except ImportError:
    print("Error: Could not import models. Ensure 'modulation_models' folder exists.")
    sys.exit(1)

try:
    from data_manager import SatComDataManager 
except ImportError:
    print("Error: data_manager.py not found.")
    sys.exit(1)

# ==============================================================================
# 1. DATASET
# ==============================================================================
class LazyIQDataset(Dataset):
    def __init__(self, data_mgr, modulations, segment_length):
        self.data_mgr = data_mgr
        self.segment_length = segment_length
        self.file_index = []
        
        if not data_mgr.master_json_path.exists():
            raise FileNotFoundError(f"Metadata not found at {data_mgr.master_json_path}")

        print(f"Loading metadata...")
        with open(data_mgr.master_json_path, 'r') as f:
            database = json.load(f)

        files = data_mgr.get_sample_list()
        mod_map = {m.lower().strip(): i for i, m in enumerate(modulations)}
        
        print(f"Scanning {len(files)} files...")
        
        for file_path in files:
            name_stem = file_path.stem
            candidate = name_stem
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
                        label = mod_map[current_mod]
                        self.file_index.append((file_path, label))

        print(f"Dataset ready. Found {len(self.file_index)} samples.")

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        file_path, label = self.file_index[idx]
        iq_data = self.data_mgr.load_iq_sample(file_path)
        
        if iq_data is None:
            return torch.zeros((2, self.segment_length), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Normalize
        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0:
            iq_data /= (max_val + 1e-6)
            
        iq_arr = np.column_stack((iq_data.real, iq_data.imag))
        
        # Pad/Crop
        if len(iq_arr) >= self.segment_length:
            segment = iq_arr[:self.segment_length]
        else:
            padding = np.zeros((self.segment_length - len(iq_arr), 2), dtype=iq_arr.dtype)
            segment = np.vstack((iq_arr, padding))
            
        # Transpose to (2, 128)
        segment = segment.transpose() 
            
        return torch.from_numpy(segment).float(), torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 2. TRAINING ROUTINE
# ==============================================================================
def train_and_eval(model, name, train_loader, test_loader, params):
    print(f"\nTraining Model: {name}")
    device = params['device']
    model.to(device)
    
    # Enable cudnn benchmark for speed consistency
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    history = []
    
    total_batches = len(train_loader)
    
    for ep in range(params['epochs']):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        t_correct, t_total = 0, 0
        running_loss = 0.0
        
        # Using enumerate to track progress without blocking
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Shape Adapter
            if "cnn" not in name.lower():
                inputs = inputs.permute(0, 2, 1) # (Batch, 128, 2)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            t_total += labels.size(0)
            t_correct += (predicted == labels).sum().item()
            
            # Print only every 10 batches to speed up loop
            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch {ep+1} | Batch {i+1}/{total_batches} | Loss: {avg_loss:.4f}", end='\r')
                running_loss = 0.0
        
        # --- VALIDATE ---
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if "cnn" not in name.lower():
                    inputs = inputs.permute(0, 2, 1)
                    
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                v_correct += (predicted == labels).sum().item()
                v_total += labels.size(0)
        
        train_acc = t_correct / t_total if t_total > 0 else 0
        val_acc = v_correct / v_total if v_total > 0 else 0
        history.append(val_acc)
        
        elapsed = time.time() - start_time
        print(f"\n >> Epoch {ep+1} Finished in {elapsed:.1f}s | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}%")

    # Save
    save_path = os.path.join(params['save_dir'], f"mod_{name.lower()}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    
    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_acc, history


# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--transformer', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    # --- CONFIGURATION ---
    params = {
        'segment_length': 128,
        'modulations': ['16APSK', '8PSK', 'QPSK', 'BPSK', '32APSK'], 
        'batch_size': 32,
        'epochs': 10,            
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': "models_results_modulation"
    }
    
    os.makedirs(params['save_dir'], exist_ok=True)
    
    if not (args.cnn or args.lstm or args.transformer or args.all):
        print("Usage: python modulation_main.py --cnn (or --lstm, --transformer, --all)")
        return

    print(f"\n--- Running on {params['device']} ---")
    
    # Data Manager
    data_mgr = SatComDataManager()
    full_dataset = LazyIQDataset(data_mgr, params['modulations'], params['segment_length'])
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # --- OPTIMIZED DATALOADERS ---
    # num_workers=2: Uses 2 background CPU cores to load files
    # pin_memory=True: Faster transfer to GPU
    # persistent_workers=True: Prevents the "Hang" after epoch 1 by keeping workers alive
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True, 
        drop_last=True, 
        num_workers=2,         
        persistent_workers=True, 
        pin_memory=True,
        prefetch_factor=2      
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=params['batch_size'], 
        shuffle=False, 
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2
    )

    results = {}
    histories = {}
    num_classes = len(params['modulations'])

    # Models definition
    models_to_run = {}
    if args.cnn or args.all:
        models_to_run["CNN"] = CNNClassifier(num_classes, params['segment_length'])
    if args.lstm or args.all:
        models_to_run["LSTM"] = LSTMClassifier(num_classes)
    if args.transformer or args.all:
        models_to_run["Transformer"] = TransformerClassifier(num_classes)

    # Execution Loop
    for name, model in models_to_run.items():
        acc, hist = train_and_eval(model, name, train_loader, test_loader, params)
        results[name] = acc
        histories[name] = hist

    # Reporting
    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v*100:.2f}%")
    
    if histories:
        plt.figure(figsize=(10, 6))
        for name, hist in histories.items():
            plt.plot(hist, label=f'{name} Val Acc')
        plt.legend()
        plt.title("Validation Accuracy")
        plt.savefig(os.path.join(params['save_dir'], "training_plot.png"))
        print("[DONE] Plot saved.")

if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()