import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split

# --- IMPORTS ---
# We keep IQDataset because it's a useful wrapper, but we do NOT use get_dataloaders (synthetic)
from modulation_models.data_modulation_utils import IQDataset 
from modulation_models.modulation_model_cnn import CNNClassifier
from modulation_models.modulation_model_lstm import LSTMClassifier
from modulation_models.modulation_model_transformer import TransformerClassifier

# Import the class from your data_manager.py
from data_manager import SatComDataManager 

def load_real_dataset_from_disk(data_mgr, modulations, segment_length):
    """
    Loads all .iq files using SatComDataManager, assigns labels based on filenames,
    and returns a PyTorch Dataset.
    """
    files = data_mgr.get_sample_list()
    
    if not files:
        raise FileNotFoundError(f"No .iq files found in {data_mgr.input_dir}")

    X_list = []
    y_list = []
    
    print(f"Processing {len(files)} files from {data_mgr.input_dir}...")
    
    files_loaded = 0
    for file_path in files:
        # 1. Determine Label from filename
        # Logic: Check if 'QPSK', '16QAM' etc is in the filename
        label = -1
        for idx, mod_name in enumerate(modulations):
            if mod_name.lower() in file_path.name.lower():
                label = idx
                break
        
        # If we can't determine the class from the filename, skip it for training
        if label == -1:
            continue

        # 2. Load Data via Manager
        iq_data = data_mgr.load_iq_sample(file_path)
        if iq_data is None: 
            continue
            
        # 3. Preprocessing (Normalization)
        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0:
            iq_data /= (max_val + 1e-6)
        
        # 4. Reshape to (Real, Imag)
        iq_arr = np.column_stack((iq_data.real, iq_data.imag))
        
        # 5. Segment into fixed lengths
        num_segments = len(iq_arr) // segment_length
        if num_segments == 0: 
            continue
            
        segments = iq_arr[:num_segments*segment_length].reshape(num_segments, segment_length, 2)
        
        X_list.append(segments)
        y_list.append(np.full((segments.shape[0],), label))
        files_loaded += 1

    if not X_list:
        raise ValueError("No labeled data could be loaded. Check filenames match modulation names.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.hstack(y_list).astype(np.int64)
    
    print(f"Data Loading Complete.")
    print(f"  Files used: {files_loaded}/{len(files)}")
    print(f"  Total Segments: {X.shape[0]}")
    print(f"  Tensor Shape: {X.shape}")
    
    return IQDataset(X, y)

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
        print(f"Epoch {ep+1}: Loss={t_loss/t_total:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

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
        # Ensure these match the strings found in your filenames exactly!
        'modulations': ['QPSK', '16QAM', '32QAM'], 
        'batch_size': 128,
        'epochs': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': "models_results_modulation"
    }
    os.makedirs(params['save_dir'], exist_ok=True)
    num_classes = len(params['modulations'])

    if not (args.cnn or args.lstm or args.transformer or args.all):
        print("Please specify a flag: --cnn, --lstm, --transformer, or --all")
        return

    # 1. Load Real Data using DataManager
    data_mgr = SatComDataManager()
    try:
        dataset = load_real_dataset_from_disk(data_mgr, params['modulations'], params['segment_length'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Split into Train/Test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], num_workers=0)

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