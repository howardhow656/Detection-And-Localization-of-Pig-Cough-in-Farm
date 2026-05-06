
import itertools
from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.io import wavfile
import numpy as np
from pathlib import Path
import json
from scipy.signal import resample_poly
import torch.nn as nn
import math



def get_mic_pairs(n_mics: int) -> List[Tuple[int, int]]:
    """產生所有 mic pair，例如 3 mic -> [(0,1), (0,2), (1,2)]"""
    return list(itertools.combinations(range(n_mics), 2))


def stft_multichannel_torch(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 1024,
) -> torch.Tensor:
    """
    多通道 STFT

    Parameters
    ----------
    audio : torch.Tensor
        shape = (n_samples, n_mics)
    Returns
    -------
    X : torch.Tensor
        shape = (n_mics, n_freq, n_frames), complex
    """
    if audio.ndim != 2:
        raise ValueError("audio shape 必須是 (n_samples, n_mics)")

    n_samples, n_mics = audio.shape
    window = torch.hann_window(win_length, device=audio.device)

    specs = []
    for m in range(n_mics):
        x = audio[:, m]
        X = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )  # (freq, frames)
        specs.append(X)

    X = torch.stack(specs, dim=0)  # (n_mics, freq, frames)
    return X


def gcc_phat_from_pair_stft(
    Xi: torch.Tensor,
    Xj: torch.Tensor,
    max_tau: int = 1024,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    對單一 mic pair 的整段 STFT 計算逐 frame GCC-PHAT

    Parameters
    ----------
    Xi, Xj : torch.Tensor
        shape = (freq, frames), complex
    Returns
    -------
    gcc : torch.Tensor
        shape = (frames, 2*max_tau+1)
    """
    if Xi.shape != Xj.shape:
        raise ValueError("Xi, Xj shape 必須相同")

    # cross spectrum: (freq, frames)
    cross_spec = Xi * torch.conj(Xj)

    # PHAT normalization
    cross_spec = cross_spec / (torch.abs(cross_spec) + eps)

    # 沿著 frequency 做 irfft -> delay domain
    # 先轉成 (frames, freq) 比較直覺
    cross_spec = cross_spec.transpose(0, 1)  # (frames, freq)

    corr = torch.fft.irfft(cross_spec, dim=-1)  # (frames, n_fft_like)
    n_delay = corr.shape[-1]

    # 把負 delay 移到左邊
    left = corr[:, -(n_delay // 2):]
    right = corr[:, : (n_delay // 2) + (n_delay % 2)]
    corr = torch.cat([left, right], dim=-1)

    tau_axis = torch.arange(
        -n_delay // 2,
        -n_delay // 2 + corr.shape[-1],
        device=corr.device
    )

    mask = (tau_axis >= -max_tau) & (tau_axis <= max_tau)
    gcc = corr[:, mask]  # (frames, 2*max_tau+1)

    return gcc


def extract_gcc_phat_feature(
    audio: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    max_tau: int = 1024,
) -> torch.Tensor:
    """
    從多通道音訊抽 GCC-PHAT feature

    Parameters
    ----------
    audio : torch.Tensor
        shape = (n_samples, n_mics)

    Returns
    -------
    feat : torch.Tensor
        shape = (n_pairs, n_frames, 2*max_tau+1)
    """
    X = stft_multichannel_torch(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )  # (n_mics, freq, frames)

    n_mics = X.shape[0]
    pairs = get_mic_pairs(n_mics)

    feat_list = []
    for i, j in pairs:
        gcc = gcc_phat_from_pair_stft(
            X[i], X[j], max_tau=max_tau
        )  # (frames, G)
        feat_list.append(gcc)

    feat = torch.stack(feat_list, dim=0)  # (pairs, frames, G)
    return feat


def slice_by_earliest_start(
    audio: ...,
    start_times,
    sample_rate: int,
    segment_sec: float = 0.5,
    pre_buffer_sec: float = 0.0
) -> torch.Tensor:
    """
    根據多支 mic 的起始時間，取最早時間作為切點，切固定長度音訊。

    Parameters
    ----------
    audio : np.ndarray or torch.Tensor
        shape = (n_samples, n_mics)
    start_times : list / tuple / np.ndarray
        每支 mic 的起始時間（單位：秒）
        例如 [1.23, 1.25, 1.28]
    sample_rate : int
        取樣率，例如 16000
    segment_sec : float
        要切出的固定長度（秒）
    pre_buffer_sec : float
        在最早起點前面額外保留多少秒，避免 onset 被切掉
        例如 0.02 表示往前多抓 20ms
    pad_mode : str
        若切到尾端不夠長時的補值方式，預設 constant = 補 0

    Returns
    -------
    segment : torch.Tensor
        shape = (segment_samples, n_mics)
    """
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, dtype=torch.float32)
    else:
        audio = audio.float()

    if audio.ndim != 2:
        raise ValueError("audio shape 必須是 (n_samples, n_mics)")

    n_samples, n_mics = audio.shape

    if len(start_times) != n_mics:
        raise ValueError(f"start_times 長度應等於 mic 數量，現在是 {len(start_times)} vs {n_mics}")

    valid_starts = [t for t in start_times if t is not None]
    if len(valid_starts) == 0:
        raise ValueError("這筆 sample 沒有任何有效的 start_time")

    earliest_start_sec = float(min(valid_starts))
    start_sec = max(0.0, earliest_start_sec - pre_buffer_sec)

    start_sample = int(round(start_sec * sample_rate))
    segment_samples = int(round(segment_sec * sample_rate))
    end_sample = start_sample + segment_samples

    if start_sample >= n_samples:
        return torch.zeros((segment_samples, n_mics), dtype=torch.float32)

    segment = audio[start_sample:min(end_sample, n_samples), :]

    if segment.shape[0] < segment_samples:
        pad_len = segment_samples - segment.shape[0]
        pad = torch.zeros((pad_len, n_mics), dtype=segment.dtype, device=segment.device)
        segment = torch.cat([segment, pad], dim=0)

    return segment

def build_pair_mask_from_start_times(start_times, n_mics: int):
    mic_valid = []

    for t in start_times:
        if t is None:
            mic_valid.append(False)
        elif isinstance(t, float) and np.isnan(t):
            mic_valid.append(False)
        else:
            mic_valid.append(True)

    pairs = get_mic_pairs(n_mics)

    pair_mask = []
    for i, j in pairs:
        pair_mask.append(1.0 if mic_valid[i] and mic_valid[j] else 0.0)

    return torch.tensor(pair_mask, dtype=torch.float32)

class GCCPhatPathDataset(Dataset):
    def __init__(
        self,
        samples,
        sample_rate=8000,
        segment_sec=0.5,
        pre_buffer_sec=0.02,
        n_fft=512,
        hop_length=512,
        win_length=2048,
        max_tau=900,
    ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.segment_sec = segment_sec
        self.pre_buffer_sec = pre_buffer_sec
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_tau = max_tau

    def __len__(self):
        return len(self.samples)

    def _load_multichannel_audio(self, audio_paths):
        wavs = []
        srs = []

        for p in audio_paths:
            sr, y = wavfile.read(p)

            # 轉 float32
            if y.dtype == np.int16:
                y = y.astype(np.float32) / 32768.0
            elif y.dtype == np.int32:
                y = y.astype(np.float32) / 2147483648.0
            elif y.dtype == np.uint8:
                y = (y.astype(np.float32) - 128) / 128.0
            else:
                y = y.astype(np.float32)

            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != self.sample_rate:
                y = resample_poly(y, self.sample_rate, sr)
                sr = self.sample_rate
            wavs.append(y)
            srs.append(sr)

        if len(set(srs)) != 1:
            raise ValueError(f"三支 mic 的 sample rate 不一致: {srs}")

        sr = srs[0]
        if sr != self.sample_rate:
            raise ValueError(f"目前程式假設 sample_rate={self.sample_rate}，但讀到 {sr}")

        min_len = min(len(y) for y in wavs)
        wavs = [y[:min_len] for y in wavs]

        audio = np.stack(wavs, axis=1)   # (n_samples, 3)
        return audio

    def __getitem__(self, idx):
        item = self.samples[idx]

        audio = self._load_multichannel_audio(item["audio_paths"])
        Position = int(item['Position'].split("-")[0]) - 1
        start_times = item["start_times"]

        segment = slice_by_earliest_start(
            audio=audio,
            start_times=start_times,
            sample_rate=self.sample_rate,
            segment_sec=self.segment_sec,
            pre_buffer_sec=self.pre_buffer_sec,
        )

        feat = extract_gcc_phat_feature(
            segment,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            max_tau=self.max_tau,
        )

        pair_mask = build_pair_mask_from_start_times(
        start_times=start_times,
        n_mics=feat.shape[0] if False else audio.shape[1]
        )
        return {
            "x": feat,
            "y": torch.tensor(Position, dtype=torch.long),
            "pair_mask": pair_mask,
        }

class CachedGCCDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

        with open(self.cache_dir / "index.json", "r", encoding="utf-8") as f:
            self.files = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], weights_only=True)

        return {
            "x": item["x"].float(),
            "pair_mask": item["pair_mask"].float(),
            "y": item["y"].long()
        }
    

def precompute_gcc_features(dataset, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    index_list = []

    for i in range(len(dataset)):
        item = dataset[i]   # 這裡會真的計算 GCC-PHAT

        x = item["x"]
        y = item["y"]
        pair_mask = item["pair_mask"]

        save_path = save_dir / f"sample_{i:06d}.pt"

        torch.save({
            "x": x.cpu(),
            "pair_mask": pair_mask.cpu(),
            "y": torch.tensor(y).long()
        }, save_path)

        index_list.append(str(save_path))

        if i % 10 == 0:
            print(f"Precomputed {i}/{len(dataset)}")

    with open(save_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_list, f, indent=4, ensure_ascii=False)

    print(f"Done. Saved {len(index_list)} samples to {save_dir}")



def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):

        x = batch["x"].to(device)   # (B, P, T, G)
        pair_mask = batch["pair_mask"].to(device)
        y = batch["y"].to(device)



        optimizer.zero_grad()
        logits = model(x, pair_mask)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)}, loss={loss.item():.4f}")
        
 
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def valid_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        y = batch["y"].to(device)

        logits = model(x, pair_mask)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total




class GCC_CRNN(nn.Module):
    def __init__(
        self,
        n_pairs: int = 3,
        gcc_bins: int = 41,
        num_classes: int = 3,
        gru_hidden: int = 64,
        dropout: float = 0.3,
        use_pair_mask: bool = True,
    ):
        super().__init__()

        self.use_pair_mask = use_pair_mask
        in_channels = n_pairs * 2 if use_pair_mask else n_pairs


        # CNN 部分：把 pair 當成 channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),   # 只壓縮 GCC 軸
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # 推導 conv 後的 GCC 維度
        reduced_gcc_bins = gcc_bins
        for _ in range(3):
            reduced_gcc_bins = math.floor(reduced_gcc_bins / 2)

        if reduced_gcc_bins < 1:
            raise ValueError("gcc_bins 太小，經過 pooling 後變 0 了")

        self.feature_dim = 128 * reduced_gcc_bins

        self.bigru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, pair_mask = None) -> torch.Tensor:
        """
        x: (B, P, T, G)
        CNN expects (B, C, H, W)
        這裡:
           C = Pairs
           H = Time
           W = GCC bins
        """

        if pair_mask is None:
            pair_mask = torch.ones(
                x.shape[0], x.shape[1],
                device=x.device,
                dtype=x.dtype
            )

        mask_map = pair_mask[:, :, None, None].expand(
            -1, -1, x.shape[2], x.shape[3]
        )

        x = torch.cat([x, mask_map], dim=1)  # (B, 2P, T, G)

        x = self.conv1(x)   # (B, 32, T, G/2)
        x = self.conv2(x)   # (B, 64, T, G/4)
        x = self.conv3(x)   # (B,128, T, G/8)

        # 轉成給 GRU 的格式：(B, T, feature_dim)
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, T, C, G')
        b, t, c, g = x.shape
        x = x.view(b, t, c * g)                  # (B, T, feature_dim)

        x, _ = self.bigru(x)                     # (B, T, 2*hidden)

        # clip-level 分類：時間平均池化
        x = x.mean(dim=1)                        # (B, 2*hidden)

        out = self.classifier(x)                # (B, num_classes)
        return out
    
class Normal_CNN(nn.Module):
    def __init__(
        self,
        n_pairs=3,
        num_classes=3,
        dropout=0.5,
        use_pair_mask=True,
    ):
        super().__init__()

        self.use_pair_mask = use_pair_mask
        in_channels = n_pairs * 2 if use_pair_mask else n_pairs

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # ⭐ 關鍵：global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, pair_mask=None):
    # x: (B, P, T, G)

            if self.use_pair_mask:
                mask_map = pair_mask[:, :, None, None].expand(
                    -1, -1, x.shape[2], x.shape[3]
                )
                x = torch.cat([x, mask_map], dim=1)

            x = self.conv(x)

            x = self.global_pool(x)  # (B, 128, 1, 1)

            out = self.classifier(x)
            return out