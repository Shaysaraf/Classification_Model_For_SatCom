import numpy as np
import os

# --------------------------------------------------------
# DVB-S1 IQ sample preparation (single file version)
# --------------------------------------------------------
# Reads a single .bin file of 16-bit interleaved IQ samples,
# splits it into chunks, labels it, and saves .npy datasets.
# --------------------------------------------------------

def load_iq_file(path, chunk_len=2048, label_name="qpsk1_4"):
    print(f"[INFO] Loading file: {path}")

    # 16-bit signed interleaved IQ
    raw = np.fromfile(path, dtype=np.int16)
    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    I /= 32768.0
    Q /= 32768.0
    iq = I + 1j * Q

    # Split into equal chunks
    n_chunks = len(iq) // chunk_len
    iq = iq[:n_chunks * chunk_len]
    iq = iq.reshape(n_chunks, chunk_len)

    # Separate real/imag as 2 channels
    X = np.stack([iq.real, iq.imag], axis=1)

    # Create label array (single class)
    y = np.full(n_chunks, 0, dtype=np.int64)

    print(f"[INFO] Created {len(X)} chunks for label '{label_name}'")
    return X, y, label_name


def main():
    filename = "dvbs1Sampler_1200M_20M_qpsk1_4.bin"
    label_name = "qpsk1_4"
    chunk_len = 2048

    # Load and process
    X, y, label_name = load_iq_file(filename, chunk_len, label_name)

    # Save dataset
    out_dir = "data_processed"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"X_{label_name}.npy"), X)
    np.save(os.path.join(out_dir, f"y_{label_name}.npy"), y)

    print(f"[DONE] Saved to {out_dir}/X_{label_name}.npy and y_{label_name}.npy")

if __name__ == "__main__":
    main()
