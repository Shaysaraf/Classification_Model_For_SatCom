import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import upfirdn

def rrc_filter(beta, sps, N=101):
    t = np.arange(-N//2, N//2+1)/sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 if beta == 0 else (1 - beta + 4*beta/np.pi)
        elif beta != 0 and abs(abs(ti) - 1/(4*beta)) < 1e-12:
            h[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            if beta == 0: h[i] = np.sin(np.pi*ti)/(np.pi*ti)
            else: h[i] = (np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))) / (np.pi*ti*(1-(4*beta*ti)**2))
    return h / np.sqrt(np.sum(h**2) + 1e-12)

def add_awgn(iq_signal, snr_db):
    sig_power = np.mean(np.abs(iq_signal)**2)
    snr_linear = 10**(snr_db/10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2)*(np.random.randn(*iq_signal.shape) + 1j*np.random.randn(*iq_signal.shape))
    return iq_signal + noise

class IQDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def get_dataloaders(beta_list, num_samples_per_beta, num_symbols, sps, snr_list, segment_length, batch_size):
    X_list, y_list = [], []
    print(f"Generating roll-off dataset for betas: {beta_list}...")
    
    for idx, beta in enumerate(beta_list):
        for _ in range(num_samples_per_beta):
            # Using QPSK as base for roll-off tests
            bits = np.random.randint(0,2,size=(num_symbols*2,))
            mapping = {(0,0):1+1j, (0,1):1-1j, (1,0):-1+1j, (1,1):-1-1j}
            symbols = np.array([mapping[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)])
            
            upsampled = upfirdn([1], symbols, up=sps)
            h = rrc_filter(beta, sps, N=101)
            iq = np.convolve(upsampled, h, mode='same')
            
            snr = np.random.choice(snr_list)
            iq = add_awgn(iq, snr)
            iq /= (np.max(np.abs(iq)) + 1e-6)
            
            iq_arr = np.column_stack((iq.real, iq.imag))
            num_segments = len(iq_arr)//segment_length
            if num_segments == 0: continue
            segments = iq_arr[:num_segments*segment_length].reshape(num_segments, segment_length, 2)
            X_list.append(segments)
            y_list.append(np.full((segments.shape[0],), idx))

    X = np.vstack(X_list).astype(np.float32)
    y = np.hstack(y_list).astype(np.int64)
    dataset = IQDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset)-train_size])
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(test_dataset, batch_size=batch_size))