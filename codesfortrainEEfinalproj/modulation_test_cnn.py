import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import psutil  # For CPU Load/Power estimation
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass

# --- FIXED MODEL IMPORT ---
from modulation_models.modulation_model_resnet18 import resnet18
from data_manager import SatComDataManager

@dataclass
class TestConfig:
    segment_length: int = 512
    # Match the lowercase tuple styling from your training configuration
    modulations: tuple = ('16apsk', '8psk', 'qpsk')
    batch_size: int = 1 
    test_split: float = 0.10 
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name: str = "best_resnet18_dpu.pth"
    metadata_path: str = r"C:\Users\shays\OneDrive\Desktop\Electrical Engineering first degree\EE 4th year\Final Project EE\codesfortrainEEfinalproj\data_ready_SR.json"
    test_data_dir: str = r"D:\iq_augmented_cut"

class SimpleTestDataset(Dataset):
    def __init__(self, file_list, database, config, data_mgr):
        self.file_list = file_list
        self.database = database
        self.config = config
        self.data_mgr = data_mgr
        self.mod_map = {m.lower().strip(): i for i, m in enumerate(config.modulations)}
        self.valid_samples = []
        
        for f in self.file_list:
            candidate = f.stem 
            entry = None
            while candidate:
                if candidate in database:
                    entry = database[candidate]
                    break
                candidate = candidate.rsplit('_', 1)[0] if '_' in candidate else None
            
            if entry:
                raw_mod = str(entry.get("modcod", "")).strip().split()[0].lower()
                if raw_mod in self.mod_map:
                    self.valid_samples.append((f, self.mod_map[raw_mod]))
        print(f"Matched {len(self.valid_samples)} valid samples.")

    def __len__(self): return len(self.valid_samples)

    def __getitem__(self, idx):
        file_path, label = self.valid_samples[idx]
        iq_data = self.data_mgr.load_iq_sample(file_path)
        if iq_data is None:
            return torch.zeros((4, self.config.segment_length)), torch.tensor(label)

        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0:
            iq_data /= (max_val + 1e-6)
            
        amp, phase = np.abs(iq_data), np.angle(iq_data)
        iq_arr = np.column_stack((iq_data.real, iq_data.imag, amp, phase))
        
        if len(iq_arr) >= self.config.segment_length:
            start = (len(iq_arr) - self.config.segment_length) // 2
            segment = iq_arr[start:start + self.config.segment_length]
        else:
            padding = np.zeros((self.config.segment_length - len(iq_arr), 4))
            segment = np.vstack((iq_arr, padding))
            
        return torch.from_numpy(segment.transpose()).float(), torch.tensor(label, dtype=torch.long)

def run_performance_test():
    config = TestConfig()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if the model is in the current execution folder or the training output directory
    model_path = os.path.join(curr_dir, config.model_name)
    if not os.path.exists(model_path):
        # Fallback to the training script's default save folder
        model_path = os.path.join(curr_dir, "models_results_resnet18", config.model_name)

    print(f"Loading weights from: {model_path}")

    with open(config.metadata_path, 'r') as f:
        database = json.load(f)

    raw_files = list(Path(config.test_data_dir).glob("*.iq"))
    if not raw_files:
        print(f"Error: No .iq files found in {config.test_data_dir}")
        return

    random.seed(42)
    random.shuffle(raw_files)
    test_files = raw_files[:max(1, int(len(raw_files) * config.test_split))]

    data_mgr = SatComDataManager()
    dataset = SimpleTestDataset(test_files, database, config, data_mgr)
    
    if len(dataset) == 0:
        print("Error: No test samples matched the specified modulation targets.")
        return
        
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # --- FIXED MODEL INSTANTIATION ---
    # Instantiating the imported ResNet18 function matching your precise Vitis-AI training configuration
    model = resnet18(
        num_classes=len(config.modulations), 
        in_channels=4, 
        input_shape=(1, config.segment_length)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device).eval()

    all_preds, all_labels, latencies, confidences = [], [], [], []
    
    print(f"Running inference and Power Profiling...")

    # Start CPU Utilization measurement
    cpu_start = psutil.cpu_percent(interval=None)

    with torch.no_grad():
        for inputs, labels in test_loader:
            start_t = time.perf_counter()
            outputs = model(inputs.to(config.device))
            latencies.append(time.perf_counter() - start_t)
            
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
            confidences.append(conf.item())
            all_preds.append(pred.item())
            all_labels.extend(labels.numpy())

    # End measurements
    cpu_end = psutil.cpu_percent(interval=None)
    avg_cpu_load = (cpu_start + cpu_end) / 2
    avg_lat = np.mean(latencies) * 1000
    avg_conf = np.mean(confidences) * 100
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    print("\n" + "="*50)
    print(f"TOTAL ACCURACY:     {accuracy:.2%}")
    print(f"AVERAGE CONFIDENCE: {avg_conf:.2f}%")
    print(f"AVG CPU LOAD:       {avg_cpu_load:.1f}%")
    print(f"AVERAGE LATENCY:    {avg_lat:.2f} ms")
    print("="*50)

    # --- ENHANCED POSTER PLOT ---
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(12, 10))
    
    # Convert labels back to uppercase just for clean presentation on your project poster
    display_labels = [m.upper() for m in config.modulations]
    
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=display_labels, yticklabels=display_labels,
                annot_kws={'size': 22, 'weight': 'bold'}) 
    
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel('Predicted Label', fontsize=16, weight='bold')
    plt.ylabel('True Label', fontsize=16, weight='bold')
    
    plt.title(f'Modulation Classification Accuracy: {accuracy:.1%}\n'
              f'Confidence: {avg_conf:.1f}% | Latency: {avg_lat:.2f}ms | CPU Load: {avg_cpu_load:.1f}%',
              fontsize=18, pad=20, weight='bold')
    
    save_path = os.path.join(curr_dir, "poster_results_enlarged.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    print(f"Poster-ready figure saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_performance_test()