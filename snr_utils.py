import numpy as np

def calculate_snr_raw(file_path):
    """
    Calculates SNR for raw IQ samples using Time Domain / Variance method.
    Auto-detects if the file is Float32 or Int16.
    """
    try:
        # 1. Try reading as Float32 (Standard)
        data = np.fromfile(file_path, dtype=np.float32)
        
        # Auto-detect if it's actually Int16 (heuristic: huge values or NaNs)
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