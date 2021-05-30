import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm import trange


def complex2real(freq_repr):
    real_D = np.zeros((freq_repr.shape[0], freq_repr.shape[1], 2), dtype=np.float32)
    real_D[:, :, 0] = np.real(freq_repr)
    real_D[:, :, 1] = np.imag(freq_repr)
    real_D = torch.from_numpy(real_D)
    return real_D


def padded_signal(y, trn_idx, hop_length):
    actual_y = []
    for idx in trn_idx:
        y_temp = y[idx*hop_length:(idx+1)*hop_length]
        if len(y_temp) < hop_length:
            rem_len = hop_length - len(y_temp)
            y_temp = np.concatenate([y_temp, np.zeros((rem_len, ))])
        actual_y.append(y_temp)
    actual_y = torch.from_numpy(np.array(actual_y, dtype=np.float32))
    return actual_y


class AudioDataSTFT:
    def __init__(self, freq_repr, time_repr, trn_idx,
                 val_idx, mode='train',
                 n_fft=2048, cvt=True):
        hop_length = n_fft//4
        if mode == 'train':
            self.time_repr = padded_signal(time_repr, trn_idx, hop_length)
            self.freq_repr = freq_repr[:, trn_idx]
        else:
            self.time_repr = padded_signal(time_repr, val_idx, hop_length)
            self.freq_repr = freq_repr[:, val_idx]
        if cvt:
            self.freq_repr = complex2real(self.freq_repr)
        self.hop_length = hop_length

    def __len__(self):
        return len(self.time_repr)

    def __getitem__(self, idx):
        return self.freq_repr[:, idx], self.time_repr[idx]


class STFTModel(nn.Module):
    def __init__(self, inp_len=2050, hop_length=512):
        super().__init__()
        self.learnt_win = nn.Parameter(torch.zeros((inp_len,), dtype=torch.float32))
        self.linear = nn.Linear(inp_len, hop_length)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.learnt_win.sigmoid() * x
        return torch.tanh(self.linear(x))


def eval_model_stft(model, loader, device='cpu'):
    y_hat = []
    full_loss = []
    with torch.no_grad():
        model.eval()
        for freq_repr, time_repr in loader:
            freq_repr = freq_repr.to(device)
            time_repr = time_repr.to(device)
            pred_y = model(freq_repr)
            full_loss.append(F.mse_loss(pred_y, time_repr).item())
            y_hat.extend(pred_y)
    y_hat = torch.hstack(y_hat).cpu().numpy()
    full_loss = np.mean(full_loss)
    return y_hat, full_loss


def train_model_stft(model, train_loader, val_loader,
                     weight_dir, fold, num_epochs=120, lr=1e-2, device='cpu'):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, mode='min',
                                                           patience=10, verbose=True)
    best_loss = float('inf')
    all_val_loss = []
    t = trange(num_epochs)
    for epoch in t:
        model.train()
        for freq_repr, time_repr in train_loader:
            freq_repr = freq_repr.to(device)
            time_repr = time_repr.to(device)
            optim.zero_grad()
            pred_time_repr = model(freq_repr)
            loss = F.mse_loss(pred_time_repr, time_repr)
            loss.backward()
            optim.step()

        _, val_loss = eval_model_stft(model, val_loader, device=device)
        all_val_loss.append(val_loss)
        t.set_postfix(val_loss=val_loss, best_loss=best_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), f"./{weight_dir}/fold_{fold}_best.pth")
            best_loss = val_loss
        scheduler.step(val_loss)
    return all_val_loss


class RAModel(nn.Module):
    def __init__(self, inp_len=2050, n_fft=2048,
                 neurons=512):
        super().__init__()
        hop_length = n_fft//4
        fft_feat_dim = (n_fft//2+1)*2

        self.linear1 = nn.Sequential(
            nn.Linear(inp_len, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, fft_feat_dim),
            nn.LeakyReLU(),
        )

        for i in range(2, 3):
            setattr(self, f"linear{i}", nn.Sequential(
                nn.Linear(fft_feat_dim, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, fft_feat_dim),
                nn.LeakyReLU(),
            ))
        self.learnt_win = nn.Parameter(torch.zeros((fft_feat_dim,), dtype=torch.float32))
        self.linear5 = nn.Linear(fft_feat_dim, hop_length)

    def forward(self, feat):
        feat = feat.reshape(feat.shape[0], -1)
        x = self.linear1(feat)
        stft = self.linear2(x) + x
        x = self.learnt_win.sigmoid() * stft
        x = torch.tanh(self.linear5(x))
        return stft, x


def eval_model(model, loader, device='cpu'):
    y_hat, all_pred_stft = [], []
    orig_stft = []
    full_loss = []
    with torch.no_grad():
        model.eval()
        for stft, freq_repr, time_repr in loader:
            freq_repr = freq_repr.to(device)
            time_repr = time_repr.to(device)
            pred_stft, pred_y = model(freq_repr)
            all_pred_stft.append(pred_stft.cpu().numpy())
            orig_stft.append(stft.cpu().numpy())
            full_loss.append(F.mse_loss(pred_y, time_repr).item())
            y_hat.extend(pred_y)
    y_hat = torch.hstack(y_hat).cpu().numpy()
    full_loss = np.mean(full_loss)
    return np.array(orig_stft), np.array(all_pred_stft), y_hat, full_loss


def train_model(model, train_loader, val_loader,
                weight_dir, fold, num_epochs=120, lr=1e-2,
                n_fft=2048, device='cpu'):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, mode='min',
                                                           patience=5, verbose=False)
    best_loss = float('inf')
    all_val_loss = []
    fft_feat_dim = (n_fft//2+1)
    t = trange(num_epochs)
    for epoch in t:
        model.train()
        for stft, freq_repr, time_repr in train_loader:
            stft = stft.to(device)
            freq_repr = freq_repr.to(device)
            time_repr = time_repr.to(device)
            optim.zero_grad()
            pred_stft, pred_time_repr = model(freq_repr)
            stft_real = F.mse_loss(pred_stft[:, :fft_feat_dim], stft[:, :fft_feat_dim])
            stft_imag = F.mse_loss(pred_stft[:, fft_feat_dim:], stft[:, fft_feat_dim:])
            stft_loss = stft_imag + stft_real
            wave_loss = 100*F.mse_loss(pred_time_repr, time_repr)
            loss = wave_loss + stft_loss
            loss.backward()
            optim.step()
        _, _, _, val_loss = eval_model(model, val_loader, device=device)
        t.set_postfix(val_loss=val_loss, best_loss=best_loss)
        all_val_loss.append(val_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), f"./{weight_dir}/fold_{fold}_best.pth")
            best_loss = val_loss
        scheduler.step(val_loss)
    torch.save(model.state_dict(), f"./{weight_dir}/fold_{fold}_last.pth")
    return all_val_loss


# # Reassigned spectrogram
class AudioData:
    def __init__(self, stft, freq_repr, time_repr, trn_idx,
                 val_idx, mode='train',
                 n_fft=2048, cvt=True):
        hop_length = n_fft//4
        if mode == 'train':
            self.time_repr = padded_signal(time_repr, trn_idx, hop_length)
            self.stft = stft[:, trn_idx]
            self.freq_repr = freq_repr[:, trn_idx]
        else:
            self.time_repr = padded_signal(time_repr, val_idx, hop_length)
            self.stft = stft[:, val_idx]
            self.freq_repr = freq_repr[:, val_idx]

        stft_mag, stft_imag = np.abs(self.stft), np.angle(self.stft)
        stft_len = self.stft.shape[0]
        self.stft = np.zeros((stft_len*2, self.stft.shape[1]), dtype=np.float32)
        self.stft[:stft_len, :] = stft_mag
        self.stft[stft_len:, :] = stft_imag
        self.hop_length = hop_length

    def __len__(self):
        return len(self.time_repr)

    def __getitem__(self, idx):
        return self.stft[:, idx], self.freq_repr[:, idx], self.time_repr[idx]


def preprocess_ra(freqs, times, mags):
    freqs, times = np.nan_to_num(freqs), np.nan_to_num(times)

    # very important!!
    # times is in global frame. need to convert it according to hop
    max_time, min_time = times.max(0)[None, :], times.min(0)[None, :]
    times = (times - min_time)/(max_time - min_time)

    max_freq, min_freq = freqs.max(0)[None, :], freqs.min(0)[None, :]
    freqs = (freqs - min_freq)/(max_freq - min_freq)

    feat = np.concatenate([freqs, times, mags], axis=0).astype(np.float32)
    return feat
