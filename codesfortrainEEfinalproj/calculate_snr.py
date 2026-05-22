import os
import json
import glob
import random
import argparse
import numpy as np
from scipy.signal import welch
from data_manager import SatComDataManager

def estimate_snr_psd(iq_samples, fs, fc_baseband, signal_bw):
    """
    Estimates the Signal-to-Noise Ratio (SNR) using the Power Spectral Density (PSD).
    """
    # Calculate PSD using Welch's method
    f, Pxx = welch(iq_samples, fs=fs, nperseg=2048, return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    
    # Identify the signal region and noise region
    signal_mask = (f >= (fc_baseband - signal_bw / 2)) & (f <= (fc_baseband + signal_bw / 2))
    noise_mask = ~signal_mask
    
    if not np.any(noise_mask):
        return 0.0
        
    # Average noise power spectral density (power per Hz)
    noise_psd = np.mean(Pxx[noise_mask])
    
    # Total power (signal + noise)
    total_power = np.mean(np.abs(iq_samples)**2)
    
    # Total noise power across the entire band
    noise_power_total = noise_psd * fs
    
    # Signal power
    signal_power = total_power - noise_power_total
    
    if signal_power <= 0:
        return -999.0
        
    snr_db = 10 * np.log10(signal_power / noise_power_total)
    return snr_db

if __name__ == "__main__":
    manager = SatComDataManager()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load JSON files
    db_path = os.path.join(curr_dir, 'data_ready_SR.json')
    mapping_path = os.path.join(curr_dir, 'class_mapping_SR.json')
    
    with open(db_path, 'r') as f:
        database = json.load(f)
        
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        
    # Invert mappings to get physical values
    inv_symbol_rate = {v: float(k) for k, v in mapping['symbol_rate'].items()}
    inv_rolloff = {v: float(k) for k, v in mapping['rolloff'].items()}
    
    # Global parameters
    fs_mhz = database.get('sampling_rate_mhz', 15)
    fs = fs_mhz * 1e6
    rf_fc_mhz = database.get('center_frequency_mhz', 1000)
    # The signal in IQ format is at baseband (0 Hz offset)
    baseband_fc = 0.0 

    parser = argparse.ArgumentParser(description="Calculate average SNR for 10% of IQ files in a directory.")
    parser.add_argument("sample_dir", type=str, help="Path to the directory containing .iq files")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    all_files = glob.glob(os.path.join(sample_dir, "*.iq"))
    
    if not all_files:
        print(f"No .iq files found in {sample_dir}")
        exit(1)
        
    num_to_sample = max(1, int(len(all_files) * 0.10))
    target_files = random.sample(all_files, num_to_sample)
    
    print(f"Found {len(all_files)} files. Sampling 10% ({num_to_sample} files) for SNR calculation.")
    print(f"Global Parameters from JSON: Sample Rate = {fs_mhz} MHz, RF Center Freq = {rf_fc_mhz} MHz\n")
    
    snr_values = []
    
    for file_path in target_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        if not os.path.exists(file_path):
            print(f"  -> File not found: {file_path}\n")
            continue
            
        iq_samples = manager.load_iq_sample(file_path)
        if iq_samples is None:
            print("  -> Failed to load IQ samples.\n")
            continue
            
        # Match file name to database key using the exact logic from modulation_test_cnn
        candidate = os.path.splitext(filename)[0]
        entry = None
        while candidate:
            if candidate in database:
                entry = database[candidate]
                break
            candidate = candidate.rsplit('_', 1)[0] if '_' in candidate else None
            
        if not entry:
            print("  -> Could not find matching entry in data_ready_SR.json!\n")
            continue
            
        # Extract parameters for this specific signal
        sr_idx = entry.get('symbol_rate')
        ro_idx = entry.get('rolloff')
        power = entry.get('power')
        modcod = entry.get('modcod')
        
        sr_mhz = inv_symbol_rate.get(sr_idx, 1.0)
        ro = inv_rolloff.get(ro_idx, 0.2)
        
        # Calculate signal bandwidth (Symbol Rate * (1 + Rolloff))
        signal_bw_mhz = sr_mhz * (1 + ro)
        signal_bw = signal_bw_mhz * 1e6
        
        print(f"  -> ModCod: {modcod}, Power level (JSON): {power} dB")
        print(f"  -> Symbol Rate: {sr_mhz} MHz, Rolloff: {ro} => Bandwidth: {signal_bw_mhz:.2f} MHz")
        
        snr_db = estimate_snr_psd(iq_samples, fs, baseband_fc, signal_bw)
        if snr_db > -900:  # ignore invalid/empty ones
            snr_values.append(snr_db)
        print(f"  -> Estimated SNR: {snr_db:.2f} dB\n")

    if snr_values:
        avg_snr = np.mean(snr_values)
        print("="*40)
        print(f"Calculated SNR for {len(snr_values)} valid samples.")
        print(f"AVERAGE SNR: {avg_snr:.2f} dB")
        print("="*40)
    else:
        print("Could not calculate SNR for any of the selected files.")
