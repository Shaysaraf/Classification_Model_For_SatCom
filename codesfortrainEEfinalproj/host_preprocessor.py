import os
import json
import numpy as np
import torch
import logging
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Adjust these raw strings to your exact paths
INPUT_DIR = Path(r"D:\iq_augmented_cut")
OUTPUT_DIR = Path(r"D:\versal_ready_bins")
METADATA_FILE = Path(r"D:\data_ready_SR.json")

# Network Parameters
SEGMENT_LENGTH = 512
MODULATIONS = ('16apsk', '8psk', 'qpsk')
TARGET_SHAPE = (1, 4, 1, SEGMENT_LENGTH) # (Batch, Channels, Height, Width)

class VersalDataBuilder:
    def __init__(self):
        self.mod_map = {m.lower().strip(): i for i, m in enumerate(MODULATIONS)}
        
        if not OUTPUT_DIR.exists():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {OUTPUT_DIR}")

    def load_raw_complex(self, file_path):
        """Robustly loads .iq or .pt files and returns a complex64 numpy array."""
        try:
            # 1. Try PyTorch .pt
            if file_path.suffix == '.pt':
                tensor_data = torch.load(file_path, map_location='cpu', weights_only=False)
                if isinstance(tensor_data, torch.Tensor):
                    return tensor_data.numpy().astype(np.complex64)
                    
            # 2. Fallback to Raw Binary
            data = np.fromfile(file_path, dtype=np.float32)
            if len(data) > 0 and (np.any(np.isnan(data)) or np.max(np.abs(data)) > 1e5):
                data = np.fromfile(file_path, dtype=np.int16)

            n_pairs = len(data) // 2
            if n_pairs == 0: 
                return None

            return (data[0:2*n_pairs:2] + 1j * data[1:2*n_pairs:2]).astype(np.complex64)
            
        except Exception:
            return None

    def process_and_save(self):
        """The main loop: Read -> Clean -> Extract -> Shape -> Save"""
        if not INPUT_DIR.exists() or not METADATA_FILE.exists():
            logging.error("Input directory or Metadata file missing. Check your paths.")
            return

        with open(METADATA_FILE, 'r') as f:
            database = json.load(f)

        files = list(INPUT_DIR.glob("*.iq")) + list(INPUT_DIR.glob("*.pt"))
        logging.info(f"Found {len(files)} candidate files. Beginning generation...")

        manifest = {}
        success_count = 0
        skip_count = 0

        for file_path in tqdm(files, desc="Building DPU Tensors"):
            # --- A. Label Matching ---
            candidate = file_path.stem
            found_entry = None
            while candidate:
                if candidate in database:
                    found_entry = database[candidate]
                    break
                if '_' in candidate: candidate = candidate.rsplit('_', 1)[0]
                else: break
                    
            if not found_entry:
                continue
                
            raw_mod = str(found_entry.get("modcod", "")).strip().split()[0].lower()
            if raw_mod not in self.mod_map:
                continue
            label = self.mod_map[raw_mod]

            # --- B. Load & Sanitize Data ---
            iq_data = self.load_raw_complex(file_path)
            if iq_data is None:
                continue
                
            # Crucial Sanity Check: Drop NaNs and Infs to prevent math crashes
            if np.isnan(iq_data).any() or np.isinf(iq_data).any():
                skip_count += 1
                continue

            # --- C. Feature Engineering (I, Q, Amp, Phase) ---
            max_val = np.max(np.abs(iq_data))
            if max_val == 0 or np.isnan(max_val):
                skip_count += 1
                continue # Skip empty signals
                
            # Normalize complex data
            iq_data = iq_data / max_val
            
            i = iq_data.real
            q = iq_data.imag
            amp = np.abs(iq_data)
            phase = np.angle(iq_data)
            
            # Stack into (4, N)
            features = np.stack([i, q, amp, phase], axis=0)

            # --- D. Crop or Pad to exactly 512 ---
            current_len = features.shape[1]
            if current_len >= SEGMENT_LENGTH:
                start = (current_len - SEGMENT_LENGTH) // 2
                features = features[:, start:start + SEGMENT_LENGTH]
            else:
                padding = np.zeros((4, SEGMENT_LENGTH - current_len), dtype=np.float32)
                features = np.concatenate([features, padding], axis=1)

            # --- E. Strictly Enforce DPU ResNet Shape (1, 4, 1, 512) ---
            final_tensor = features.reshape(TARGET_SHAPE)

            # --- F. Dump to C-Contiguous Binary ---
            out_name = f"{file_path.stem}.bin"
            out_path = OUTPUT_DIR / out_name
            
            contiguous_tensor = np.ascontiguousarray(final_tensor, dtype=np.float32)
            contiguous_tensor.tofile(out_path)
            
            # Log to manifest
            manifest[out_name] = {
                "label": label, 
                "mod": raw_mod,
                "original_file": file_path.name
            }
            success_count += 1

        # --- G. Save Manifest ---
        with open(OUTPUT_DIR / "versal_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        logging.info(f"Done! Successfully generated {success_count} ready-to-use DPU tensors.")
        if skip_count > 0:
            logging.warning(f"Skipped {skip_count} corrupted/empty files.")

if __name__ == "__main__":
    builder = VersalDataBuilder()
    builder.process_and_save()