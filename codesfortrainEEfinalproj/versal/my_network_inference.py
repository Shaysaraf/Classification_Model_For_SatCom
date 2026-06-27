import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Flat imports matching your exact directory layout
from modulation_model_resnet18 import resnet18
from data_manager import SatComDataManager

# --- CONFIGURATION ---
SEGMENT_LENGTH = 512
MODULATIONS = ('16apsk', '8psk', 'qpsk')
SEED = 42
BATCH_SIZE = 1  
MODEL_WEIGHTS_PATH = "best_resnet18_dpu.pth"  # Looks directly in the current folder
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LazyIQDataset(torch.utils.data.Dataset):
    """
    Exact replica of your training LazyIQDataset to ensure preprocessing matches 
    the floating-point model's expected execution environment.
    """
    def __init__(self, data_mgr, modulations, segment_length, mode='eval'):
        self.data_mgr = data_mgr
        self.segment_length = segment_length
        self.mode = mode 
        self.file_index = []
        
        if not data_mgr.master_json_path.exists():
            raise FileNotFoundError(f"Metadata not found at {data_mgr.master_json_path}")

        with open(data_mgr.master_json_path, 'r') as f:
            database = json.load(f)

        files = sorted(data_mgr.get_sample_list())
        mod_map = {m.lower().strip(): i for i, m in enumerate(modulations)}
        
        for file_path in files:
            candidate = file_path.stem
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
                        self.file_index.append((file_path, mod_map[current_mod]))

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx):
        file_path, label = self.file_index[idx]
        iq_data = self.data_mgr.load_iq_sample(file_path)
        
        if iq_data is None:
            return torch.zeros((4, self.segment_length)), torch.tensor(label)

        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0: 
            iq_data /= (max_val + 1e-6)
            
        if self.mode == 'train':
            theta = np.random.uniform(0, 2 * np.pi)
            iq_data = iq_data * np.exp(1j * theta)

        amp = np.abs(iq_data)
        phase = np.angle(iq_data)
        iq_arr = np.column_stack((iq_data.real, iq_data.imag, amp, phase))
        
        if len(iq_arr) >= self.segment_length:
            if self.mode == 'train':
                start = np.random.randint(0, len(iq_arr) - self.segment_length + 1)
            else:
                start = (len(iq_arr) - self.segment_length) // 2
            segment = iq_arr[start:start + self.segment_length]
        else:
            padding = np.zeros((self.segment_length - len(iq_arr), 4), dtype=iq_arr.dtype)
            segment = np.vstack((iq_arr, padding))
            
        segment = segment.transpose() 
        return torch.from_numpy(segment).float(), torch.tensor(label, dtype=torch.long)


def main():
    print(f"--> Initializing Inference Environment on device: {DEVICE}")
    
    # 1. Initialize Data Manager and seamlessly target local folder paths
    data_mgr = SatComDataManager()
    data_mgr.input_dir = Path("iq_augmented_cut")
    data_mgr.master_json_path = Path("data_ready_SR.json")
    data_mgr.output_dir = Path("results")
    
    # 2. Setup Evaluation Dataset
    eval_dataset = LazyIQDataset(data_mgr, MODULATIONS, SEGMENT_LENGTH, mode='eval')
    total_size = len(eval_dataset)
    
    if total_size == 0:
        raise RuntimeError("No dataset samples found. Verify 'iq_augmented_cut' and 'data_ready_SR.json' exist here.")

    # 3. Slice exactly 10% of the entire dataset for calibration samples
    calib_size = max(1, int(0.10 * total_size))
    
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(total_size, generator=generator).tolist()
    calib_ds = Subset(eval_dataset, indices[:calib_size])
    
    data_loader = DataLoader(calib_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"--> Local dataset mapped. Total available: {total_size} samples.")
    print(f"--> Extracted exactly 10% ({calib_size} samples) for Vitis-AI calibration.")

    # 4. Model Setup & Weights Loading
    num_classes = len(MODULATIONS)
    model = resnet18(
        num_classes=num_classes, 
        in_channels=4, 
        input_shape=(1, SEGMENT_LENGTH)
    )
    
    if Path(MODEL_WEIGHTS_PATH).exists():
        print(f"--> Loading model checkpoint from current directory: {MODEL_WEIGHTS_PATH}")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    else:
        print(f"[WARNING] Checkpoint '{MODEL_WEIGHTS_PATH}' not found! Running with random weights.")

    model = model.to(DEVICE)
    model.eval()

    # 5. Inference & Calibration Loop
    correct = 0
    total = 0
    
    print("\n--> Starting Calibration Stream...")
    with torch.no_grad():
        for i, (in_data, labels) in enumerate(data_loader):
            in_data = in_data.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(in_data)
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if i % 10 == 0 or i < 5:
                print(f"Process Sample Index: {i:04d}/{calib_size} | Accuracy: {(preds == labels).sum().item()/labels.size(0):.0%}")

    # 6. Summary Output
    final_accuracy = (correct / total) * 100 if total > 0 else 0
    print("\n" + "="*50)
    print(f"CALIBRATION INFERENCE RUN COMPLETE")
    print(f"Total Processed (10% Split): {total}")
    print(f"Average Model Prediction Accuracy: {final_accuracy:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()