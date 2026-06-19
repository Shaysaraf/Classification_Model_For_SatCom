import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# IMPORTANT: Import your exact 2D ResNet18 architecture from your model file
from modulation_models.modulation_model_resnet18 import resnet18

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = Path(r"D:\versal_ready_bins")
# Wrapped in Path() and ready for your NEW 2D trained weights
MODEL_WEIGHTS = Path(r"D:\best_resnet18_dpu.pth") 
BATCH_SIZE = 64
SEGMENT_LENGTH = 512
MODULATIONS = ('16apsk', '8psk', 'qpsk')

# Auto-detect hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. DATASET DEFINITION
# ==========================================
class VersalDPUDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "versal_manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}. Did you run the preprocessor?")
            
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.filenames = list(self.manifest.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = self.data_dir / filename
        label = self.manifest[filename]["label"]
        
        # Read the C-Contiguous float32 binary file
        data_array = np.fromfile(filepath, dtype=np.float32).reshape(4, 1, SEGMENT_LENGTH)
        
        return torch.from_numpy(data_array), torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
def run_evaluation():
    print(f"--- Starting Evaluation on {DEVICE} ---")
    
    if not DATA_DIR.exists():
        print(f"[!] Error: Data directory {DATA_DIR} not found.")
        return
    if not MODEL_WEIGHTS.exists():
        print(f"[!] Error: Weights file {MODEL_WEIGHTS.resolve()} not found. Ensure you have retrained the 2D model.")
        return

    # 1. Load Data
    dataset = VersalDPUDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} testing samples.")

    # 2. Initialize Model
    model = resnet18(
        num_classes=len(MODULATIONS), 
        in_channels=4, 
        input_shape=(1, SEGMENT_LENGTH)
    )
    
    # Load weights safely
    state_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    # 3. Run Inference
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Calculate Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean() * 100.0
    print(f"\n=========================================")
    print(f" TOTAL ACCURACY: {accuracy:.2f}%")
    print(f"=========================================\n")
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=MODULATIONS))

    # 5. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=MODULATIONS, yticklabels=MODULATIONS)
    plt.title("ResNet18 Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = "evaluation_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"[+] Saved confusion matrix to: {os.path.abspath(cm_path)}")

if __name__ == "__main__":
    run_evaluation()