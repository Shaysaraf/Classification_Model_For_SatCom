import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Paths - Adjust these to match your PC/Jetson environment
INPUT_DIR = Path(os.environ.get('DATA_DIR', r'D:\iq_augmented_cut'))
OUTPUT_DIR = Path("versal_ready_bins")
METADATA_FILE = Path(os.environ.get('METADATA_FILE', 'data_ready_SR.json'))

# Model Parameters
SEGMENT_LENGTH = 512
MODULATIONS = ('16apsk', '8psk', 'qpsk')

# ==========================================
# ROBUST .IQ LOADER
# ==========================================
def load_iq_sample(file_path):
    """
    Reads IQ data from either a PyTorch archive OR a raw binary .iq file.
    Replicates the logic from SatComDataManager.
    """
    try:
        # METHOD 1: Try loading as PyTorch Tensor
        try:
            tensor_data = torch.load(file_path, map_location='cpu')
            if isinstance(tensor_data, torch.Tensor):
                return tensor_data.numpy().astype(np.complex64)
        except Exception:
            pass # Not a torch file, fallback to raw binary

        # METHOD 2: Fallback to Raw Binary (Standard .iq)
        data = np.fromfile(file_path, dtype=np.float32)

        # Check if it looks like Int16 (Heuristic: massive values or NaNs)
        if len(data) > 0 and (np.any(np.isnan(data)) or np.max(np.abs(data)) > 1e5):
            data = np.fromfile(file_path, dtype=np.int16)

        # Convert to Complex
        n_pairs = len(data) // 2
        if n_pairs == 0:
            return None

        i_samples = data[0:2*n_pairs:2].astype(np.float32)
        q_samples = data[1:2*n_pairs:2].astype(np.float32)
        signal_complex = i_samples + 1j * q_samples
        
        return signal_complex
        
    except Exception as e:
        print(f"[!] Failed to read file {file_path}: {e}")
        return None

# ==========================================
# PREPROCESSING LOOP
# ==========================================
def build_versal_dataset():
    if not INPUT_DIR.exists():
        print(f"[!] Input directory not found: {INPUT_DIR}")
        return
        
    if not METADATA_FILE.exists():
        print(f"[!] Metadata file not found: {METADATA_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        database = json.load(f)
        
    mod_map = {m.lower().strip(): i for i, m in enumerate(MODULATIONS)}
    manifest = {}
    
    # Find all .iq files
    files = list(INPUT_DIR.glob("*.iq"))
    print(f"Found {len(files)} .iq files to process in {INPUT_DIR}")
    
    processed_count = 0
    
    for file_path in tqdm(files, desc="Processing signals for Versal DPU"):
        # 1. Label Matching
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
                
        if not found_entry:
            continue
            
        raw_mod = str(found_entry.get("modcod", "")).strip().split()[0].lower()
        if raw_mod not in mod_map:
            continue
            
        label = mod_map[raw_mod]
        
        # 2. Data Extraction
        iq_data = load_iq_sample(file_path)
        if iq_data is None:
            continue
            
        # 3. Normalization
        iq_data = iq_data.astype(np.complex64)
        max_val = np.max(np.abs(iq_data))
        if max_val > 0: 
            iq_data /= (max_val + 1e-6)
            
        # 4. Feature Engineering (I, Q, Amplitude, Phase)
        amp = np.abs(iq_data)
        phase = np.angle(iq_data)
        iq_arr = np.column_stack((iq_data.real, iq_data.imag, amp, phase))
        
        # 5. Cropping / Padding (Evaluation mode logic: Center Crop)
        if len(iq_arr) >= SEGMENT_LENGTH:
            start = (len(iq_arr) - SEGMENT_LENGTH) // 2
            segment = iq_arr[start:start + SEGMENT_LENGTH]
        else:
            padding = np.zeros((SEGMENT_LENGTH - len(iq_arr), 4), dtype=iq_arr.dtype)
            segment = np.vstack((iq_arr, padding))
            
        # 6. Transpose to NCHW Format -> Shape becomes (4, 512)
        segment = segment.transpose()
        
        # 7. ENFORCE C-CONTIGUOUS MEMORY (CRITICAL FOR VERSAL AIE)
        # Bypasses the invalid strides ONNX runtime error on hardware
        segment_contiguous = np.ascontiguousarray(segment, dtype=np.float32)
        
        # 8. Dump to Raw Binary
        out_name = f"{file_path.stem}.bin"
        out_path = OUTPUT_DIR / out_name
        segment_contiguous.tofile(out_path)
        
        # 9. Update Manifest
        manifest[out_name] = {
            "label": label, 
            "mod": raw_mod,
            "original_file": file_path.name
        }
        processed_count += 1
        
    # Export the manifest for the edge evaluation script
    manifest_path = OUTPUT_DIR / "versal_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\n=== PREPROCESSING COMPLETE ===")
    print(f"Successfully processed {processed_count} files.")
    print(f"Output Directory: {OUTPUT_DIR.resolve()}")
    print("Action Required: Transfer the 'versal_ready_bins' folder to the Versal Edge platform.")

if __name__ == "__main__":
    build_versal_dataset()