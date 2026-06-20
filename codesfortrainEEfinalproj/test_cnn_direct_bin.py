import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Import the architecture function
from modulation_models.modulation_model_resnet18 import resnet18

# ==========================================
# 1. CONFIGURATION
# ==========================================
BIN_DIR = Path(r"D:\versal_ready_bins")
MODEL_PATH = Path(r"D:\best_resnet18_dpu.pth")

SEGMENT_LENGTH = 512
BATCH_SIZE = 64  # Can increase batch size since pre-processing overhead is gone
MODULATIONS = ('16apsk', '8psk', 'qpsk')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. BINARY LOADER DATASET
# ==========================================
class VersalBinDataset(Dataset):
    def __init__(self, bin_dir):
        self.bin_dir = Path(bin_dir)
        manifest_path = self.bin_dir / "versal_manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing at {manifest_path}. Run Script 1 first!")
            
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.filenames = list(self.manifest.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = self.bin_dir / filename
        label = self.manifest[filename]["label"]
        
        # Pull flat array bytes and restore the exact 3D channel layout
        data_array = np.fromfile(filepath, dtype=np.float32).reshape(4, 1, SEGMENT_LENGTH)
        return torch.from_numpy(data_array).float(), torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. PERFORMANCE EVALUATION RUNNER
# ==========================================
def evaluate_preprocessed_bins():
    print(f"--- Starting Preprocessed Binary Evaluation on {DEVICE} ---")
    
    if not BIN_DIR.exists():
        print(f"[!] Error: Preprocessed binary path {BIN_DIR} doesn't exist.")
        return

    # 1. Prepare Data
    dataset = VersalBinDataset(BIN_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Successfully loaded {len(dataset)} binary files for testing.")

    # 2. Instantiate Model mirroring exact training specs
    model = resnet18(
        num_classes=len(MODULATIONS), 
        in_channels=4, 
        input_shape=(1, SEGMENT_LENGTH)
    )
    
    print(f"Loading weights from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    all_preds, all_labels, latencies, confidences = [], [], [], []
    
    # Profile hardware loads
    cpu_start = psutil.cpu_percent(interval=None)

    # 3. Evaluate loop
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="DPU Binary Inference"):
            start_t = time.perf_counter()
            outputs = model(inputs.to(DEVICE))
            latencies.append(time.perf_counter() - start_t)
            
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
            confidences.extend(conf.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    cpu_end = psutil.cpu_percent(interval=None)
    
    # 4. Compute Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    avg_lat = (np.mean(latencies) / BATCH_SIZE) * 1000  # Per-sample latency normalized by batch
    avg_conf = np.mean(confidences) * 100
    avg_cpu_load = (cpu_start + cpu_end) / 2

    print("\n" + "="*50)
    print(f"TOTAL BINARY ACCURACY: {accuracy:.2%}")
    print(f"AVERAGE CONFIDENCE:    {avg_conf:.2f}%")
    print(f"AVG CPU LOAD:          {avg_cpu_load:.1f}%")
    print(f"AMORTIZED LATENCY:     {avg_lat:.2f} ms/sample")
    print("="*50 + "\n")
    
    print("Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=MODULATIONS))

    # 5. Generate Enhanced Poster Plot
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(12, 10))
    
    display_labels = [m.upper() for m in MODULATIONS]
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=display_labels, yticklabels=display_labels,
                annot_kws={'size': 22, 'weight': 'bold'}) 
    
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel('Predicted Label', fontsize=16, weight='bold')
    plt.ylabel('True Label', fontsize=16, weight='bold')
    
    plt.title(f'Binary Dataset Validation Accuracy: {accuracy:.1%}\n'
              f'Confidence: {avg_conf:.1f}% | Latency: {avg_lat:.2f}ms | CPU: {avg_cpu_load:.1f}%',
              fontsize=18, pad=20, weight='bold')
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binary_validation_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    print(f"[+] Saved execution plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_preprocessed_bins()