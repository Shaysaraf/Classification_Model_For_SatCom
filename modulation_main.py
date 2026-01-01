import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
import matplotlib.pyplot as plt
import modulation_models as models

# --- IMPORTS MUST MATCH FILE NAMES EXACTLY ---
from modulation_models.data_modulation_utils import get_dataloaders
from modulation_models.modulation_model_cnn import CNNClassifier
from modulation_models.modulation_model_lstm import LSTMClassifier
from modulation_models.modulation_model_transformer import TransformerClassifier

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
        
        train_acc = t_correct / t_total
        val_acc = v_correct / v_total
        history.append(train_acc)
        print(f"Epoch {ep+1}: Loss={t_loss/t_total:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(params['save_dir'], f"mod_{name.lower()}.pth"))
    
    # Explicit Cleanup for Jetson
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_acc, history

def main():
    parser = argparse.ArgumentParser(description="Train Modulation Classification Models")
    parser.add_argument('--cnn', action='store_true', help='Train the CNN model')
    parser.add_argument('--lstm', action='store_true', help='Train the LSTM model')
    parser.add_argument('--transformer', action='store_true', help='Train the Transformer model')
    parser.add_argument('--all', action='store_true', help='Train all models')
    args = parser.parse_args()

    # Configuration
    params = {
        'num_symbols': 512,
        'sps': 8,
        'segment_length': 128,
        'num_samples_per_mod': 2000,
        'modulations': ['QPSK', '16QAM', '32QAM'],
        'snr_list': [5, 10, 15, 20, 30],
        'batch_size': 128,
        'epochs': 10,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': "models_results_modulation"
    }
    os.makedirs(params['save_dir'], exist_ok=True)
    num_classes = len(params['modulations'])

    # 1. Prepare Data (Only if we are actually training something)
    if not (args.cnn or args.lstm or args.transformer or args.all):
        print("Please specify a flag: --cnn, --lstm, --transformer, or --all")
        return

    train_loader, test_loader = get_dataloaders(
        params['modulations'], params['num_samples_per_mod'], params['num_symbols'],
        params['sps'], params['snr_list'], params['segment_length'], params['batch_size']
    )
    
    results = {}
    histories = {}

    # 2. Logic for Selecting Models
    models_to_run = {}
    if args.cnn or args.all:
        models_to_run["CNN"] = CNNClassifier(num_classes, params['segment_length'])
    if args.lstm or args.all:
        models_to_run["LSTM"] = LSTMClassifier(num_classes)
    if args.transformer or args.all:
        models_to_run["Transformer"] = TransformerClassifier(num_classes)

    # 3. Execution Loop
    for name, model in models_to_run.items():
        acc, hist = train_and_eval(model, name, train_loader, test_loader, params)
        results[name] = acc
        histories[name] = hist

    print("\nFinal Test Accuracies:", results)
    
    # 4. Optional Plot
    if histories:
        plt.figure(figsize=(10, 6))
        for name, hist in histories.items():
            plt.plot(hist, label=f'{name} Train Acc')
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(params['save_dir'], "training_plot.png"))
        print(f"Plot saved to {params['save_dir']}/training_plot.png")

if __name__ == "__main__":
    main()