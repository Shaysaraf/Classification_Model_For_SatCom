import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Parameters
# -----------------------------------
filename = "dvbs1Sampler_1200M_20M_qpsk1_4.bin"
sample_rate = 40e6                      # 40 Msps
num_samples_plot = 8000                 # Visible part of the waveform
fft_size = 2**18                        # FFT length (~260k)

# -----------------------------------
# Load 16-bit signed interleaved IQ
# -----------------------------------
raw = np.fromfile(filename, dtype=np.int16)

# Convert interleaved -> complex
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)

print("Loaded", len(iq), "complex samples")

# Normalize to [-1, 1]
iq /= 32768.0


# ============================================================
# 1) PLOT I AND Q ON THE SAME AXES
# ============================================================
plt.figure(figsize=(12,5))
plt.plot(np.real(iq[:num_samples_plot]), label='I')
plt.plot(np.imag(iq[:num_samples_plot]), label='Q')
plt.title("I and Q (Same Plot)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 2) PLOT I AND Q IN SEPARATE SUBPLOTS
# ============================================================
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(np.real(iq[:num_samples_plot]), color='blue')
plt.title("In-phase Component (I)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(np.imag(iq[:num_samples_plot]), color='red')
plt.title("Quadrature Component (Q)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 3) AMPLITUDE (MAGNITUDE) PLOT
# ============================================================
amp = np.abs(iq)

plt.figure(figsize=(12,4))
plt.plot(amp[:num_samples_plot])
plt.title("Amplitude (|IQ|)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 4) PHASE PLOT
# ============================================================
phase = np.angle(iq)

plt.figure(figsize=(12,4))
plt.plot(phase[:num_samples_plot])
plt.title("Phase (radians)")
plt.xlabel("Sample Index")
plt.ylabel("Phase [rad]")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 5) FFT PLOT (Spectrum)
# ============================================================
N = min(fft_size, len(iq))
window = np.hanning(N)

iq_win = iq[:N] * window
fft_data = np.fft.fftshift(np.fft.fft(iq_win))
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))

# Convert to dB
fft_db = 20*np.log10(np.abs(fft_data) + 1e-12)

plt.figure(figsize=(12,5))
plt.plot(freqs/1e6, fft_db)
plt.title("Spectrum (FFT)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()
# ============================================================
# 6) CONSTELLATION PLOT
# ============================================================

# Select a subset for clarity
num_const_points = 5000
iq_subset = iq[:num_const_points]

plt.figure(figsize=(6,6))
plt.scatter(np.real(iq_subset), np.imag(iq_subset), color='blue', s=5, alpha=0.6)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("QPSK Constellation Diagram")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.axis('equal')
plt.show()
