
import itertools
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np


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
    max_tau: int = 20,
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
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 1024,
    max_tau: int = 20,
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
    audio: np.ndarray | torch.Tensor,
    start_times,
    sample_rate: int,
    segment_sec: float = 0.5,
    pre_buffer_sec: float = 0.0,
    pad_mode: str = "constant",
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

    earliest_start_sec = float(min(start_times))
    start_sec = max(0.0, earliest_start_sec - pre_buffer_sec)

    start_sample = int(round(start_sec * sample_rate))
    segment_samples = int(round(segment_sec * sample_rate))
    end_sample = start_sample + segment_samples

    if start_sample >= n_samples:
        # 起點超出音檔尾端，直接回傳全 0
        return torch.zeros((segment_samples, n_mics), dtype=torch.float32)

    segment = audio[start_sample:min(end_sample, n_samples), :]

    # 如果尾端不夠長，補 0
    if segment.shape[0] < segment_samples:
        pad_len = segment_samples - segment.shape[0]
        pad = torch.zeros((pad_len, n_mics), dtype=segment.dtype, device=segment.device)
        segment = torch.cat([segment, pad], dim=0)

    return segment



class GCCPhatPathDataset(Dataset):
    def __init__(
        self,
        samples,
        sample_rate=16000,
        segment_sec=0.5,
        pre_buffer_sec=0.02,
        n_fft=512,
        hop_length=160,
        win_length=512,
        max_tau=20,
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
            y, sr = sf.read(p, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
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
        label = item["label"]
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

        return {
            "x": feat,
            "y": torch.tensor(label, dtype=torch.long),
        }






def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device)   # (B, P, T, G)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

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
        y = batch["y"].to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total