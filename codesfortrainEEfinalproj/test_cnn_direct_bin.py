import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import your specific CNN architecture
from modulation_models.modulation_model_cnn import CNNClassifier

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "versal_ready_bins" 
MANIFEST_PATH = os.path.join(DATA_DIR, "versal_manifest.json")
MODEL_WEIGHTS = "best_resnet_cnn.pth"
BATCH_SIZE = 64
MODULATIONS = ('16APSK', '8PSK', 'QPSK')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# ZERO-PREPROCESSING DATASET
# ==========================================
class RawBinaryDataset(Dataset):
    """
    Bypasses SatComDataManager entirely.
    Reads pre-formatted C-contiguous float32 binaries directly from disk.
    """
    def __init__(self, data_dir, manifest_path):
        self.data_dir = data_dir
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.file_list = list(self.manifest.keys())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.data_dir, filename)
        
        # 1. READ PURE BINARY - NO MATH, NO OVERHEAD
        # Expected shape from host_preprocessor: (4, 512)
        signal_array = np.fromfile(filepath, dtype=np.float32).reshape(4, 512)
        
        # 2. Extract Ground Truth Label from Manifest
        label = self.manifest[filename]["label"]
        
        return torch.from_numpy(signal_array), torch.tensor(label, dtype=torch.long)

# ==========================================
# EVALUATION LOOP
# ==========================================
def evaluate_cnn():
    print(f"--- Direct Binary PyTorch Evaluation ---")
    print(f"Target Device: {DEVICE}")
    print(f"Loading weights from: {MODEL_WEIGHTS}")

    # 1. Initialize Dataset and Dataloader
    dataset = RawBinaryDataset(DATA_DIR, MANIFEST_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Found {len(dataset)} pre-processed signals.")

    # 2. Initialize Model
    # Assuming standard initialization. Adjust parameters if your CNNClassifier __init__ differs.
    model = CNNClassifier(num_classes=len(MODULATIONS)) 
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    except Exception as e:
        print(f"[!] Failed to load model weights. Ensure '{MODEL_WEIGHTS}' is in the directory.")
        print(f"Error: {e}")
        return

    model.to(DEVICE)
    model.eval()

    # 3. Metrics Tracking
    all_preds = []
    all_labels = []
    total_time = 0.0

    # 4. Inference Loop
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating Network"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            start_time = time.perf_counter()
            outputs = model(inputs)
            end_time = time.perf_counter()
            
            total_time += (end_time - start_time)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Compute Statistics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean() * 100.0
    avg_latency_ms = (total_time / len(dataset)) * 1000.0

    print("\n" + "="*50)
    print("               EVALUATION RESULTS               ")
    print("="*50)
    print(f"TOTAL ACCURACY:  {accuracy:.2f}%")
    print(f"AVERAGE LATENCY: {avg_latency_ms:.4f} ms per sample")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=MODULATIONS))

    # 6. Plot Confusion Matrix (Matched to your poster design style)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=MODULATIONS, yticklabels=MODULATIONS,
                annot_kws={'size': 18, 'weight': 'bold'})
    
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel('Predicted Label', fontsize=16, weight='bold')
    plt.ylabel('True Label', fontsize=16, weight='bold')
    plt.title(f'Direct Binary Validation Accuracy: {accuracy:.2f}%', fontsize=18, weight='bold')
    
    plt.tight_layout()
    plt.savefig("binary_validation_confusion_matrix.png", dpi=300)
    print("\n[+] Saved confusion matrix to 'binary_validation_confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_cnn()
    