import os
import json
import numpy as np
import logging
import torch
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
INPUT_DIR = r"D:\iq_augmented_cut"
OUTPUT_DIR = r"D:\versal_ready_bins"
METADATA_PATH = r"C:\Users\shays\OneDrive\Desktop\Electrical Engineering first degree\EE 4th year\Final Project EE\codesfortrainEEfinalproj\data_ready_SR.json"

SEGMENT_LENGTH = 512
MODULATIONS = ('16apsk', '8psk', 'qpsk')
TARGET_SHAPE = (1, 4, 1, SEGMENT_LENGTH) # Strict DPU format

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 2. YOUR EXACT SATCOM DATA MANAGER
# ==========================================
class SatComDataManager:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.extension = ".iq"

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_sample_list(self):
        if not self.input_dir.exists():
            return []
        return list(self.input_dir.glob(f"*{self.extension}")) + list(self.input_dir.glob("*.pt"))

    def load_iq_sample(self, file_path):
        try:
            try:
                tensor_data = torch.load(file_path, map_location='cpu', weights_only=False)
                if isinstance(tensor_data, torch.Tensor):
                    return tensor_data.numpy().astype(np.complex64)
            except Exception:
                pass

            data = np.fromfile(file_path, dtype=np.float32)
            if len(data) > 0 and (np.any(np.isnan(data)) or np.max(np.abs(data)) > 1e5):
                data = np.fromfile(file_path, dtype=np.int16)

            n_pairs = len(data) // 2
            if n_pairs == 0:
                return None

            i_samples = data[0:2*n_pairs:2].astype(np.float32)
            q_samples = data[1:2*n_pairs:2].astype(np.float32)
            return i_samples + 1j * q_samples
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            return None

# ==========================================
# 3. PREPROCESSING & MANIFEST GENERATION
# ==========================================
def build_dataset():
    data_mgr = SatComDataManager(INPUT_DIR, OUTPUT_DIR)
    mod_map = {m.lower().strip(): i for i, m in enumerate(MODULATIONS)}
    
    with open(METADATA_PATH, 'r') as f:
        database = json.load(f)
        
    raw_files = data_mgr.get_sample_list()
    logging.info(f"Found {len(raw_files)} total raw samples. Processing...")
    
    manifest = {}
    success_count = 0

    for file_path in tqdm(raw_files, desc="Processing Tensors"):
        # Match label using your custom rsplit loop logic
        candidate = file_path.stem
        entry = None
        while candidate:
            if candidate in database:
                entry = database[candidate]
                break
            candidate = candidate.rsplit('_', 1)[0] if '_' in candidate else None
            
        if not entry:
            continue
            
        raw_mod = str(entry.get("modcod", "")).strip().split()[0].lower()
        if raw_mod not in mod_map:
            continue
        label = mod_map[raw_mod]

        # Load samples identically
        iq_data = data_mgr.load_iq_sample(file_path)
        if iq_data is None:
            continue

        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0:
            iq_data /= (max_val + 1e-6)
            
        amp, phase = np.abs(iq_data), np.angle(iq_data)
        iq_arr = np.column_stack((iq_data.real, iq_data.imag, amp, phase))
        
        # Center-slice window logic matching your dataset
        if len(iq_arr) >= SEGMENT_LENGTH:
            start = (len(iq_arr) - SEGMENT_LENGTH) // 2
            segment = iq_arr[start:start + SEGMENT_LENGTH]
        else:
            padding = np.zeros((SEGMENT_LENGTH - len(iq_arr), 4))
            segment = np.vstack((iq_arr, padding))
            
        features = segment.transpose().astype(np.float32) # Shape: (4, 512)
        final_tensor = features.reshape(TARGET_SHAPE)     # Shape: (1, 4, 1, 512)

        # Export as clean C-contiguous flat floats
        out_name = f"{file_path.stem}.bin"
        out_path = Path(OUTPUT_DIR) / out_name
        final_tensor.tofile(out_path)
        
        manifest[out_name] = {
            "label": label,
            "mod": raw_mod
        }
        success_count += 1

    with open(Path(OUTPUT_DIR) / "versal_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    logging.info(f"Successfully generated {success_count} aligned binary tensors at {OUTPUT_DIR}")

if __name__ == "__main__":
    build_dataset()