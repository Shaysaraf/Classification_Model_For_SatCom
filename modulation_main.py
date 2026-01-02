import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split

# --- IMPORTS ---
from modulation_models.modulation_model_cnn import CNNClassifier
from modulation_models.modulation_model_lstm import LSTMClassifier
from modulation_models.modulation_model_transformer import TransformerClassifier
from data_manager import SatComDataManager 

# --- NEW LAZY DATASET CLASS ---
class LazyIQDataset(Dataset):
    """
    Loads IQ data on-the-fly to save RAM.
    Pre-scans the directory to build an index of valid files and their labels.
    """
    def __init__(self, data_mgr, modulations, segment_length):
        self.data_mgr = data_mgr
        self.segment_length = segment_length
        self.file_index = []  # List of (file_path, label) tuples
        
        # Load metadata once
        if not data_mgr.master_json_path.exists():
            raise FileNotFoundError(f"Master metadata file not found at {data_mgr.master_json_path}")
        
        print(f"Loading metadata from {data_mgr.master_json_path}...")
        with open(data_mgr.master_json_path, 'r') as f:
            database = json.load(f)

        files = data_mgr.get_sample_list()
        print(f"Indexing {len(files)} files (this may take a moment)...")
        
        valid_count = 0
        
        for file_path in files:
            # 1. Determine Label from JSON
            name_only = file_path.stem 
            parts = name_only.split('_')
            
            mod_str = None
            if len(parts) > 1:
                key = "_".join(parts[:-1])
                # Safe lookup, handle cases where mod might be missing or int
                if key in database:
                    entry = database[key]
                    if isinstance(entry, dict):
                        mod_str = str(entry.get("mod", "")).strip()

            label = -1
            if mod_str:
                for idx, target_mod in enumerate(modulations):
                    if target_mod.lower() == mod_str.lower():
                        label = idx
                        break
            
            if label != -1:
                self.file_index.append((file_path, label))
                valid_count += 1

        print(f"Indexing complete. Found {valid_count} valid samples.")
        if valid_count == 0:
            raise ValueError("No valid labeled files found. Check JSON/filenames.")

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        file_path, label = self.file_index[idx]
        
        # Load data on demand
        iq_data = self.data_mgr.load_iq_sample(file_path)
        
        # If read fails (corrupt file), return a zero tensor (or handle gracefully)
        # Note: In PyTorch DataLoader, returning None usually crashes custom collate_fns.
        # Ideally, we filter these out beforehand, but for lazy loading we might return zeros.
        if iq_data is None:
            return torch.zeros((self.segment_length, 2), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Preprocessing (Normalization)
        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0:
            iq_data /= (max_val + 1e-6)
        
        # Reshape to (Real, Imag)
        iq_arr = np.column_stack((iq_data.real, iq_data.imag))
        
        # Truncate or Pad to segment_length
        # (This logic differs slightly from previous code which took MULTIPLE segments per file.
        # For simple classification, we usually take the FIRST segment or a random crop.
        # Here we take the first segment to keep 1-to-1 mapping with file index.)
        if len(iq_arr) >= self.segment_length:
            segment = iq_arr[:self.segment_length]
        else:
            # Pad if too short
            padding = np.zeros((self.segment_length - len(iq_arr), 2), dtype=iq_arr.dtype)
            segment = np.vstack((iq_arr, padding))
            
        return torch.from_numpy(segment).float(), torch.tensor(label, dtype=torch.long)

def train_and_eval(model, name, train_loader, test_loader, params):
    print(f"\n--- Training {name} ---")
    model.to(params['device'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    for ep in range(params['epochs']):
        # TRAIN
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        
        # Added simple progress print for long epochs
        batch_count = 0
        total_batches = len(train_loader)
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(params['device']), labels.to(params['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * inputs.size(0)
            t_correct += outputs.max(1)[1].eq(labels).sum().item()
            t_total += labels.size(0)
            
            batch_count += 1
            if batch_count % 100 == 0:
                print(f"  Epoch {ep+1} [{batch_count}/{total_batches}]", end='\r')
        
        # VALIDATE
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
        print(f"\nEpoch {ep+1}: Loss={t_loss/t_total:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(params['save_dir'], f"mod_{name.lower()}.pth"))
    
    # Cleanup
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_acc, history

def main():
    parser = argparse.ArgumentParser(description="Train Modulation Classification Models on Real Data")
    parser.add_argument('--cnn', action='store_true', help='Train the CNN model')
    parser.add_argument('--lstm', action='store_true', help='Train the LSTM model')
    parser.add_argument('--transformer', action='store_true', help='Train the Transformer model')
    parser.add_argument('--all', action='store_true', help='Train all models')
    args = parser.parse_args()

    # Configuration
    params = {
        'segment_length': 128,
        'modulations': ['QPSK', '16QAM', '32QAM'], 
        'batch_size': 64,  # Reduced batch size to help memory
        'epochs': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': "models_results_modulation"
    }
    os.makedirs(params['save_dir'], exist_ok=True)
    num_classes = len(params['modulations'])

    if not (args.cnn or args.lstm or args.transformer or args.all):
        print("Please specify a flag: --cnn, --lstm, --transformer, or --all")
        return

    # 1. Load Real Data Index (Lazy Loading)
    data_mgr = SatComDataManager()
    try:
        # We replace the old load function with our class instantiation
        dataset = LazyIQDataset(data_mgr, params['modulations'], params['segment_length'])
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    # 2. Split into Train/Test
    # Note: random_split works fine with lazy datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # num_workers > 0 allows background loading, which is great for lazy loading
    # BUT on Jetson, too many workers can also cause OOM. Start with 2 or 4.
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], num_workers=2)

    results = {}
    histories = {}

    # 3. Model Selection
    models_to_run = {}
    if args.cnn or args.all:
        models_to_run["CNN"] = CNNClassifier(num_classes, params['segment_length'])
    if args.lstm or args.all:
        models_to_run["LSTM"] = LSTMClassifier(num_classes)
    if args.transformer or args.all:
        models_to_run["Transformer"] = TransformerClassifier(num_classes)

    # 4. Training Loop
    for name, model in models_to_run.items():
        acc, hist = train_and_eval(model, name, train_loader, test_loader, params)
        results[name] = acc
        histories[name] = hist

    print("\nFinal Test Accuracies:", results)
    
    # 5. Plotting
    if histories:
        plt.figure(figsize=(10, 6))
        for name, hist in histories.items():
            plt.plot(hist, label=f'{name} Train Acc')
        plt.title("Training Accuracy on Real Data")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(params['save_dir'], "training_plot.png"))
        print(f"Plot saved to {params['save_dir']}/training_plot.png")

if __name__ == "__main__":
    main()