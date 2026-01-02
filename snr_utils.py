import json
import os
import numpy as np

def calculate_snr_raw(file_path):
    """
    Calculates SNR for raw IQ samples using Time Domain / Variance method.
    """
    try:
        # 1. Load data
        data = np.fromfile(file_path, dtype=np.float32)
        
        # Auto-detect if it's actually int16
        if len(data) > 0 and (np.any(np.isnan(data)) or np.max(np.abs(data)) > 1e5):
            data = np.fromfile(file_path, dtype=np.int16)

        # 2. Handle pairing and convert to complex
        n_pairs = len(data) // 2
        if n_pairs == 0:
            return 0.0
            
        i_samples = data[0:2*n_pairs:2].astype(np.float64)
        q_samples = data[1:2*n_pairs:2].astype(np.float64)
        signal_complex = i_samples + 1j * q_samples
        
        # 3. SNR Estimation
        total_power = np.mean(np.abs(signal_complex)**2)
        mag = np.abs(signal_complex)
        noise_est = np.var(mag) 
        sig_est = total_power - noise_est
        
        if sig_est <= 0:
            return 0.0
            
        snr_db = 10 * np.log10(sig_est / (noise_est + 1e-12))
        return round(float(snr_db), 2)

    except Exception:
        return 0.0

def main():
    input_json = 'data_ready_SR.json'
    output_json = 'data_for_train.json'
    
    if not os.path.exists(input_json):
        print(f"CRITICAL: {input_json} not found!")
        return

    with open(input_json, 'r') as f:
        database = json.load(f)

    new_dataset = {}
    iq_files = [f for f in os.listdir('.') if f.endswith('.iq')]
    
    if not iq_files:
        print("No .iq files found.")
        return

    print(f"Found {len(iq_files)} files. Processing...")

    for filename in iq_files:
        # "1_2_0.1_20_1.iq" -> lookup key "1_2_0.1_20"
        name_only = os.path.splitext(filename)[0]
        parts = name_only.split('_')
        
        if len(parts) > 1:
            key = "_".join(parts[:-1])
            
            if key in database:
                snr_val = calculate_snr_raw(filename)
                params = database[key]
                
                # --- NEW OUTPUT FORMAT ---
                new_dataset[filename] = {
                    "mod": params.get("mod"),
                    "rolloff": params.get("rolloff"),
                    "snr_measured": snr_val
                }
                print(f"Processed: {filename} -> {snr_val} dB")

    # Write the new formatted JSON
    with open(output_json, 'w') as f:
        json.dump(new_dataset, f, indent=4)
    
    print(f"\nSuccessfully created {output_json} with the requested format.")

if __name__ == "__main__":
    main()