import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import upfirdn

def add_awgn(iq_signal, snr_db):
    sig_power = np.mean(np.abs(iq_signal)**2)
    snr_linear = 10**(snr_db/10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2)*(np.random.randn(*iq_signal.shape) + 1j*np.random.randn(*iq_signal.shape))
    return iq_signal + noise

def generate_symbols(mod, num_symbols):
    if mod == 'QPSK':
        bits = np.random.randint(0, 2, size=(num_symbols*2,))
        mapping = {(0,0):1+1j, (0,1):1-1j, (1,0):-1+1j, (1,1):-1-1j}
        return np.array([mapping[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)])
    else:
        M = int(mod.replace('QAM',''))
        k = int(np.log2(M))
        bits = np.random.randint(0,2,size=(num_symbols*k,))
        symbols_idx = []
        for i in range(0, len(bits), k):
            idx = 0
            for b in range(k):
                idx = (idx << 1) | int(bits[i+b])
            symbols_idx.append(idx)
        symbols_idx = np.array(symbols_idx)
        m = int(np.sqrt(M))
        I = 2*(symbols_idx % m) - m + 1
        Q = 2*(symbols_idx // m) - m + 1
        return I + 1j*Q

class IQDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def get_dataloaders(modulations, num_samples_per_mod, num_symbols, sps, snr_list, segment_length, batch_size):
    X_list, y_list = [], []
    print(f"Generating modulation dataset for: {modulations}...")
    
    for idx, mod in enumerate(modulations):
        for _ in range(num_samples_per_mod):
            symbols = generate_symbols(mod, num_symbols)
            iq = upfirdn([1], symbols, up=sps)
            snr = np.random.choice(snr_list)
            iq = add_awgn(iq, snr)
            iq /= (np.max(np.abs(iq)) + 1e-6)
            iq_arr = np.column_stack((iq.real, iq.imag))
            num_segments = len(iq_arr) // segment_length
            if num_segments == 0: continue
            segments = iq_arr[:num_segments*segment_length].reshape(num_segments, segment_length, 2)
            X_list.append(segments)
            y_list.append(np.full((segments.shape[0],), idx))

    X = np.vstack(X_list).astype(np.float32)
    y = np.hstack(y_list).astype(np.int64)
    print(f"Dataset generation complete. Total shape: {X.shape}")

    dataset = IQDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # num_workers=0 is safer for Jetson RAM
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    return train_loader, test_loader