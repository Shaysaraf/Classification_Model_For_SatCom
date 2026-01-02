import os
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Configure logging to see what's happening on the remote terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DataManager] - %(levelname)s - %(message)s'
)

class SatComDataManager:
    def __init__(self):
        # --- HARDCODED PATHS AS REQUESTED ---
        self.input_dir = Path("/mnt/usb/iq_augmented_cut")
        self.output_dir = Path("/mnt/usb/amir_and_shay_results")
        self.extension = ".iq"

        # Ensure the output directory exists on the HDD
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created output directory: {self.output_dir}")
            except Exception as e:
                logging.error(f"Could not create output dir on HDD: {e}")

    def get_sample_list(self):
        """
        Scans the input HDD directory for all .iq files.
        Returns: A list of Path objects.
        """
        if not self.input_dir.exists():
            logging.error(f"Input directory not found: {self.input_dir}")
            return []

        # Find all files matching the extension
        files = list(self.input_dir.glob(f"*{self.extension}"))
        
        if not files:
            logging.warning(f"No {self.extension} files found in {self.input_dir}")
        else:
            logging.info(f"Found {len(files)} samples ready for processing.")
            
        return files

    def load_iq_sample(self, file_path):
        """
        Reads a binary .iq file from the HDD into a numpy array.
        Assumes Complex64 (Standard for GNURadio/SDR).
        """
        try:
            # reading raw binary data
            # Adjust 'dtype' if your data is float32 interleaved or int16
            data = np.fromfile(file_path, dtype=np.complex64)
            return data
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            return None

    def save_inference_result(self, original_file_path, results_dict):
        """
        Saves the network output to the HDD as a JSON file.
        
        Args:
            original_file_path (Path): The path of the input .iq file.
            results_dict (dict): The output from your network (e.g., {'class': 'QPSK', 'snr': 12})
        """
        # Create a filename: sample_01.iq -> sample_01_prediction.json
        output_filename = original_file_path.stem + "_prediction.json"
        save_path = self.output_dir / output_filename
        
        # Add timestamp to the result data
        results_dict["processed_at"] = datetime.now().isoformat()
        results_dict["source_file"] = str(original_file_path.name)

        try:
            with open(save_path, 'w') as f:
                json.dump(results_dict, f, indent=4)
            # logging.info(f"Saved: {output_filename}") # Uncomment if you want spam in logs
        except Exception as e:
            logging.error(f"Failed to save results for {original_file_path.name}: {e}")

# ==========================================
# Example: How to integrate this into your Main Loop
# ==========================================
if __name__ == "__main__":
    
    # 1. Initialize Manager
    data_mgr = SatComDataManager()
    
    # 2. Get all files from HDD
    iq_files = data_mgr.get_sample_list()

    # 3. Processing Loop
    print("Starting Batch Processing...")
    
    for iq_file in iq_files:
        # A. Load Data (HDD -> RAM)
        iq_data = data_mgr.load_iq_sample(iq_file)
        
        if iq_data is None:
            continue

        # --- NETWORK INFERENCE HERE ---
        # inputs = preprocess(iq_data)
        # outputs = model(inputs)
        
        # Mocking a result for demonstration
        mock_output = {
            "modulation": "QPSK",
            "confidence": 0.98,
            "snr_db": 14.5,
            "is_anomaly": False
        }
        # ------------------------------

        # B. Save Result (RAM -> HDD)
        data_mgr.save_inference_result(iq_file, mock_output)

    print("Processing Complete. Check /mnt/usb/amir_and_shay_results")