import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------
# Parameters & Folder Setup
# -----------------------------------
iq_folder = "converted_iq_files" # The folder containing your .iq files
sample_rate = 40e6               # 40 Msps
num_samples_plot = 8000          # Visible part of the waveform
fft_size = 2**18                 # FFT length
num_const_points = 5000          # Points for constellation

def analyze_file(filepath, filename):
    print(f"Analyzing: {filename}")
    
    # Load complex samples (assuming complex64 from your conversion script)
    iq = np.fromfile(filepath, dtype=np.complex64)
    
    # Optional: Normalize to [-1, 1] if not already done in conversion
    mag_max = np.max(np.abs(iq))
    if mag_max > 0:
        iq /= mag_max

    # ============================================================
    # PLOT 1: TIME DOMAIN (I/Q, Amp, Phase)
    # ============================================================
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f"Time Domain Analysis: {filename}", fontsize=16)

    # I and Q
    axs[0].plot(np.real(iq[:num_samples_plot]), label='I', alpha=0.8)
    axs[0].plot(np.imag(iq[:num_samples_plot]), label='Q', alpha=0.8)
    axs[0].set_title("In-phase and Quadrature")
    axs[0].grid(True)
    axs[0].legend()

    # Amplitude
    axs[1].plot(np.abs(iq[:num_samples_plot]), color='green')
    axs[1].set_title("Amplitude Over Time")
    axs[1].grid(True)

    # Phase
    axs[2].plot(np.angle(iq[:num_samples_plot]), color='purple')
    axs[2].set_title("Phase (radians)")
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ============================================================
    # PLOT 2: SPECTRUM & CONSTELLATION
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # FFT / Spectrum
    N = min(fft_size, len(iq))
    window = np.hanning(N)
    iq_win = iq[:N] * window
    fft_data = np.fft.fftshift(np.fft.fft(iq_win))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))
    fft_db = 20*np.log10(np.abs(fft_data) + 1e-12)

    ax1.plot(freqs/1e6, fft_db)
    ax1.set_title(f"Spectrum (FFT) - {filename}")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True)

    # Constellation
    iq_subset = iq[:num_const_points]
    ax2.scatter(np.real(iq_subset), np.imag(iq_subset), color='blue', s=2, alpha=0.5)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_title("Constellation Diagram")
    ax2.set_xlabel("I")
    ax2.set_ylabel("Q")
    ax2.axis('equal')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# -----------------------------------
# Main Loop
# -----------------------------------
if __name__ == "__main__":
    if not os.path.exists(iq_folder):
        print(f"Error: Folder '{iq_folder}' not found.")
    else:
        # Filter for .iq files
        files = [f for f in os.listdir(iq_folder) if f.lower().endswith(".iq")]
        
        if not files:
            print("No .iq files found in the directory.")
        else:
            for f in files:
                file_path = os.path.join(iq_folder, f)
                analyze_file(file_path, f)

