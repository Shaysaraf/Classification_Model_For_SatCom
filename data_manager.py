import os
import json
import numpy as np
import logging
import torch  # Required for loading .pt files
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DataManager] - %(levelname)s - %(message)s'
)

class SatComDataManager:
    def __init__(self):
        # --- PATHS ---
        # Update these paths if your mount point changes
        self.input_dir = Path("/mnt/usb/iq_augmented_cut")
        self.output_dir = Path("/mnt/usb/amir_and_shay_results")
        self.master_json_path = Path("data_ready_SR.json") # Restored this line
        self.extension = ".iq"

        # Ensure the output directory exists
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created output directory: {self.output_dir}")
            except Exception as e:
                logging.error(f"Could not create output dir on HDD: {e}")

    def get_sample_list(self):
        """
        Returns a list of path objects for all .iq files in the input directory.
        """
        if not self.input_dir.exists():
            logging.warning(f"Input directory not found: {self.input_dir}")
            return []
            
        files = list(self.input_dir.glob(f"*{self.extension}"))
        if not files:
            logging.warning(f"No {self.extension} files found in {self.input_dir}")
        else:
            logging.info(f"Found {len(files)} files in {self.input_dir}")
            
        return files

    def load_iq_sample(self, file_path):
        """
        Reads IQ data from either a PyTorch (.pt/.iq) archive OR a raw binary file.
        Returns: Numpy array (Complex64) or None if failed.
        """
        try:
            # --- METHOD 1: Try loading as PyTorch Tensor ---
            try:
                # map_location='cpu' ensures it loads even if saved on a different GPU
                tensor_data = torch.load(file_path, map_location='cpu')
                
                if isinstance(tensor_data, torch.Tensor):
                    # Convert to numpy complex64
                    return tensor_data.numpy().astype(np.complex64)
            except Exception:
                # Not a torch file, proceed to Method 2
                pass

            # --- METHOD 2: Fallback to Raw Binary (Standard .iq) ---
            # Try Float32 first
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
            logging.error(f"Failed to read file {file_path}: {e}")
            return None

    def save_inference_result(self, original_file_path, results_dict):
        """
        Saves network output/predictions to HDD as a JSON file.
        """
        output_filename = Path(original_file_path).stem + "_prediction.json"
        save_path = self.output_dir / output_filename
        
        # Add metadata
        results_dict["processed_at"] = datetime.now().isoformat()
        results_dict["source_file"] = str(Path(original_file_path).name)

        try:
            with open(save_path, 'w') as f:
                json.dump(results_dict, f, indent=4)
            logging.info(f"Saved results to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")

# =========================================================
#  Standalone Execution: Test the Data Loading
# =========================================================
if __name__ == "__main__":
    print("--- SatCom Data Manager ---")
    
    manager = SatComDataManager()
    
    # 1. Get list of files
    all_files = manager.get_sample_list()
    
    if all_files:
        # 2. Test load the first file
        test_file = all_files[0]
        print(f"\nAttempting to load: {test_file.name}")
        
        data = manager.load_iq_sample(test_file)
        
        if data is not None:
            print(f"Successfully loaded data.")
            print(f"Shape: {data.shape}")
            print(f"Type: {data.dtype}")
            print(f"First 5 samples: {data[:5]}")
            
            # 3. Test saving a dummy result
            dummy_result = {"prediction": "QPSK", "confidence": 0.98}
            manager.save_inference_result(test_file, dummy_result)
        else:
            print("Failed to load data.")
    else:
        print("No files found to test.")