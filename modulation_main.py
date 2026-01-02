import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- IMPORTS ---
try:
    from modulation_models.modulation_model_cnn import CNNClassifier
    from modulation_models.modulation_model_lstm import LSTMClassifier
    from modulation_models.modulation_model_transformer import TransformerClassifier
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Ensure you have the 'modulation_models' folder with the correct __init__.py and model files.")
    exit(1)

from data_manager import SatComDataManager 

# ==============================================================================
# 1. ROBUST LAZY DATASET CLASS
# ==============================================================================
class LazyIQDataset(Dataset):
    def __init__(self, data_mgr, modulations, segment_length):
        self.data_mgr = data_mgr
        self.segment_length = segment_length
        self.file_index = []
        
        # Load JSON
        if not data_mgr.master_json_path.exists():
            raise FileNotFoundError(f"Master metadata file not found at {data_mgr.master_json_path}")

        print(f"Loading metadata from {data_mgr.master_json_path}...")
        with open(data_mgr.master_json_path, 'r') as f:
            database = json.load(f)

        files = data_mgr.get_sample_list()
        print(f"Scanning {len(files)} files...")
        
        # Create mapping based on your specific order:
        # ['16APSK', '8PSK', 'QPSK'] -> {'16apsk': 0, '8psk': 1, 'qpsk': 2}
        mod_map = {m.lower().strip(): i for i, m in enumerate(modulations)}
        
        valid_count = 0
        
        for file_path in files:
            name_stem = file_path.stem
            candidate = name_stem
            found_entry = None
            
            # --- PEELING LOGIC (Matches "1_2_0.1_20_1" to "1_2_0.1_20") ---
            while candidate:
                if candidate in database:
                    found_entry = database[candidate]
                    break
                if '_' in candidate:
                    candidate = candidate.rsplit('_', 1)[0]
                else:
                    break
            # -------------------------------------------------------------

            if found_entry:
                # Parse "modcod": "QPSK 1/4" -> "QPSK"
                raw_mod = str(found_entry.get("modcod", "")).strip()
                
                if raw_mod:
                    current_mod = raw_mod.split()[0].lower() # e.g. "qpsk"
                    
                    if current_mod in mod_map:
                        label = mod_map[current_mod]
                        self.file_index.append((file_path, label))
                        valid_count += 1

        print(f"Indexing complete. Found {valid_count} valid samples for classes: {modulations}")
        
        if valid_count == 0:
            print("\n--- DEBUG FAILURE ---")
            print(f"Target Modulations: {list(mod_map.keys())}")
            print("No matching files found. Check if your JSON 'modcod' values match the target list.")
            raise ValueError("No valid labeled files found.")

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        file_path, label = self.file_index[idx]
        
        # Load IQ Data
        iq_data = self.data_mgr.load_iq_sample(file_path)
        
        # Handle Corrupt Data
        if iq_data is None:
            return torch.zeros((self.segment_length, 2), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Preprocessing
        iq_data = iq_data.astype(np.complex64)
        
        # Normalize
        max_val = np.max(np.abs(iq_data))
        if max_val > 0:
            iq_data /= (max_val + 1e-6)
            
        # Reshape to (Segment_Length, 2)
        iq_arr = np.column_stack((iq_data.real, iq_data.imag))
        
        # Pad or Crop
        if len(iq_arr) >= self.segment_length:
            segment = iq_arr[:self.segment_length]
        else:
            padding = np.zeros((self.segment_length - len(iq_arr), 2), dtype=iq_arr.dtype)
            segment = np.vstack((iq_arr, padding))
            
        return torch.from_numpy(segment).float(), torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 2. TRAINING LOOP (With ETA & Acc)
# ==============================================================================
def train_and_eval(model, name, train_loader, test_loader, params):
    print(f"\n========================================")
    print(f"   TRAINING MODEL: {name}")
    print(f"========================================")
    
    model.to(params['device'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    
    for ep in range(params['epochs']):
        start_time = time.time()
        
        # --- TRAIN PHASE ---
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        
        batch_count = 0
        total_batches = len(train_loader)
        
        for inputs, labels in train_loader:
            batch_start = time.time()
            inputs, labels = inputs.to(params['device']), labels.to(params['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Stats
            t_loss += loss.item() * inputs.size(0)
            t_correct += outputs.max(1)[1].eq(labels).sum().item()
            t_total += labels.size(0)
            
            batch_count += 1
            
            # Print Progress & ETA
            if batch_count % 10 == 0 or batch_count == total_batches:
                batch_dur = time.time() - batch_start
                eta = (total_batches - batch_count) * batch_dur
                current_acc = t_correct / t_total
                print(f"  Epoch {ep+1} | Batch {batch_count}/{total_batches} | "
                      f"Acc: {current_acc:.4f} | ETA: {eta:.1f}s   ", end='\r')
        
        # --- VALIDATION PHASE ---
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(params['device']), labels.to(params['device'])
                outputs = model(inputs)
                v_correct += outputs.max(1)[1].eq(labels).sum().item()
                v_total += labels.size(0)
        
        train_acc = t_correct / t_total if t_total > 0 else 0
        val_acc = v_correct / v_total if v_total > 0 else 0
        history.append(train_acc)
        
        epoch_dur = time.time() - start_time
        print(f"\n  >> Epoch {ep+1} Finished ({epoch_dur:.1f}s): "
              f"Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Loss={t_loss/t_total:.4f}")

    # --- SAVE ---
    save_path = os.path.join(params['save_dir'], f"mod_{name.lower()}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  [SAVED] {save_path}")
    
    # --- CLEANUP ---
    del model
    del optimizer
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
        # EXACT ORDER YOU REQUESTED: 0=16APSK, 1=8PSK, 2=QPSK
        'modulations': ['16APSK', '8PSK', 'QPSK'], 
        'batch_size': 64,       
        'epochs': 10,           
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': "models_results_modulation"
    }
    
    os.makedirs(params['save_dir'], exist_ok=True)
    num_classes = len(params['modulations'])

    if not (args.cnn or args.lstm or args.transformer or args.all):
        print("\nUsage Error: Specify a model (e.g., --all)")
        return

    # 1. DATASET
    print(f"\n--- Initializing Dataset on {params['device']} ---")
    data_mgr = SatComDataManager()
    
    try:
        full_dataset = LazyIQDataset(data_mgr, params['modulations'], params['segment_length'])
    except ValueError as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        return

    # 2. SPLIT
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Total Samples: {len(full_dataset)}")
    print(f"Training Set:  {len(train_dataset)}")
    print(f"Testing Set:   {len(test_dataset)}")

    # 3. LOADERS
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2)

    results = {}
    histories = {}

    # 4. MODELS
    models_to_run = {}
    if args.cnn or args.all:
        models_to_run["CNN"] = CNNClassifier(num_classes, params['segment_length'])
    if args.lstm or args.all:
        models_to_run["LSTM"] = LSTMClassifier(num_classes)
    if args.transformer or args.all:
        models_to_run["Transformer"] = TransformerClassifier(num_classes)

    # 5. RUN
    for name, model in models_to_run.items():
        acc, hist = train_and_eval(model, name, train_loader, test_loader, params)
        results[name] = acc
        histories[name] = hist

    # 6. RESULTS
    print("\n========================================")
    print("   FINAL ACCURACIES")
    print("========================================")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc*100:.2f}%")
    
    if histories:
        plt.figure(figsize=(10, 6))
        for name, hist in histories.items():
            plt.plot(hist, label=f'{name} Train Acc')
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(params['save_dir'], "training_comparison.png"))
        print(f"\n[DONE] Plot saved.")

if __name__ == "__main__":
    main()