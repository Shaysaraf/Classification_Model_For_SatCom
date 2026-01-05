import os
import json
import numpy as np
import logging
import torch  # <--- REQUIRED for your specific files
from pathlib import Path
from datetime import datetime

# --- IMPORT YOUR UTILS ---
try:
    from snr_utils import calculate_snr_raw
except ImportError:
    logging.error("CRITICAL: 'snr_utils.py' not found. Ensure it is in the same directory.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DataManager] - %(levelname)s - %(message)s'
)

class SatComDataManager:
    def __init__(self):
        # --- PATHS ---
        self.input_dir = Path("/mnt/usb/iq_augmented_cut")
        self.output_dir = Path("/mnt/usb/amir_and_shay_results")
        # Path to your master lookup table
        self.master_json_path = Path("data_ready_SR.json") 
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
            
        return files

    def load_iq_sample(self, file_path):
        """
        Reads IQ data from either a PyTorch (.pt/.iq) archive OR a raw binary file.
        """
        try:
            # --- METHOD 1: Try loading as PyTorch Tensor (Fixes your specific error) ---
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

    def create_training_dataset(self, output_json_name='data_for_train.json'):
        """
        Scans all files, calculates SNR using snr_utils, looks up labels, 
        and saves the final JSON for training.
        """
        if not self.master_json_path.exists():
            logging.error(f"Master JSON not found at {self.master_json_path}")
            return

        logging.info("Loading master database...")
        with open(self.master_json_path, 'r') as f:
            database = json.load(f)

        if not self.input_dir.exists():
            logging.error(f"Input directory not found: {self.input_dir}")
            return

        iq_files = list(self.input_dir.glob(f"*{self.extension}"))
        if not iq_files:
            logging.warning(f"No files found in {self.input_dir}")
            return

        logging.info(f"Found {len(iq_files)} files. Building dataset...")
        
        new_dataset = {}
        
        for idx, file_path in enumerate(iq_files):
            filename = file_path.name
            
            # Show progress every 100 files
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(iq_files)}...", end='\r')

            # Parse filename: "1_2_0.1_20_1.iq" -> Key: "1_2_0.1_20"
            name_only = file_path.stem 
            parts = name_only.split('_')
            
            if len(parts) > 1:
                key = "_".join(parts[:-1])
                
                if key in database:
                    params = database[key]
                    
                    # --- CALL SNR UTILS HERE ---
                    # We pass the full path as a string
                    snr_val = calculate_snr_raw(str(file_path))
                    
                    new_dataset[filename] = {
                        "mod": params.get("mod"),
                        "rolloff": params.get("rolloff"),
                        "snr_measured": snr_val
                    }
        
        print(f"\nProcessing complete. Valid samples: {len(new_dataset)}")

        # Save result to HDD
        save_path = self.output_dir / output_json_name
        try:
            with open(save_path, 'w') as f:
                json.dump(new_dataset, f, indent=4)
            logging.info(f"Saved training dataset to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON: {e}")

    def save_inference_result(self, original_file_path, results_dict):
        """
        Saves network output to HDD.
        """
        output_filename = original_file_path.stem + "_prediction.json"
        save_path = self.output_dir / output_filename
        
        results_dict["processed_at"] = datetime.now().isoformat()
        results_dict["source_file"] = str(original_file_path.name)

        try:
            with open(save_path, 'w') as f:
                json.dump(results_dict, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save results: {e}")

# =========================================================
#  Standalone Execution: Run this to generate the dataset
# =========================================================
if __name__ == "__main__":
    print("--- SatCom Data Manager ---")
    
    manager = SatComDataManager()
    
    # Run the dataset creation process
    manager.create_training_dataset()