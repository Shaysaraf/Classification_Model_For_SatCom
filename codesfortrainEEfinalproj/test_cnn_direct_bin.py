import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import your specific ResNet18 architecture
from modulation_models.modulation_model_resnet18 import resnet18

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = r"D:\versal_ready_bins" 
MODEL_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_resnet_cnn.pth")
MANIFEST_PATH = os.path.join(DATA_DIR, "versal_manifest.json")
BATCH_SIZE = 64
SEGMENT_LENGTH = 512
MODULATIONS = ('16apsk', '8psk', 'qpsk') 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RawBinaryDataset(Dataset):
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
        
        # Read pre-formatted (4, 512) binary
        signal_array = np.fromfile(filepath, dtype=np.float32).reshape(4, 512)
        label = self.manifest[filename]["label"]
        
        return torch.from_numpy(signal_array), torch.tensor(label, dtype=torch.long)

def evaluate_resnet():
    print(f"--- Evaluating ResNet18 on: {DEVICE} ---")
    print(f"Loading weights: {MODEL_WEIGHTS}")

    dataset = RawBinaryDataset(DATA_DIR, MANIFEST_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Instantiate with the exact training parameters
    model = resnet18(
        num_classes=len(MODULATIONS), 
        in_channels=4, 
        input_shape=(1, SEGMENT_LENGTH)
    )
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"[!] Error: {MODEL_WEIGHTS} not found.")
        return

    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            
            # ResNet-18 typically expects (Batch, Channels, Height, Width)
            # If your model expects 2D images, we unsqueeze to add the height dimension
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(2) 
                
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Statistics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100.0
    print(f"\nTOTAL ACCURACY: {accuracy:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=MODULATIONS))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=MODULATIONS, yticklabels=MODULATIONS)
    plt.savefig("validation_results.png")
    print("[+] Saved confusion matrix to 'validation_results.png'")

if __name__ == "__main__":
    evaluate_resnet()