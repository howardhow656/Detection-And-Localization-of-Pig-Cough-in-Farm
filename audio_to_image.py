import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import soundfile as sf 
import librosa
import scipy.signal as signal
import json



with open(r'model_config.json', "r", encoding="utf-8") as f:
    config = json.load(f)



def spectrogram(y ,sr ,hop_length):
    MIN_N = 2048
    if len(y) < MIN_N:
        y = np.pad(y, (0, MIN_N - len(y)), mode='constant')
    D = np.abs(librosa.stft(y,hop_length=hop_length)) ** 2
    return D
def log_mel(y, sr, n_mels, hop_length):
    """
    回傳 log-mel spectrogram：(n_mels, T)
    不再做 mfcc 的 DCT，而是直接用 log-mel 當特徵。
    """
    D = spectrogram(y, sr, hop_length)  # power spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        S=D, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel_spectrogram)  # 轉成 dB
    return log_mel


def mfcc(y, sr, n_mels, hop_length):
    D = spectrogram(y, sr, hop_length)
    mel_spectrogram = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel_spectrogram)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=n_mels)
    return mfcc

def bandpass_filter(
    y,
    sr,
    low_cut=config['bandpass']['low_cut'],
    high_cut=config['bandpass']['high_cut'],
    order=8,
    zero_phase=True,  
):
    """
    穩健的帶通濾波：
    - 使用 SOS（butter(..., output='sos')）
    - 若訊號太短無法 sosfiltfilt，就改用 sosfilt（不報錯）
    - 自動夾住截止頻率在 (0, Nyquist) 內
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  

    nyq = 0.5 * float(sr)

    low = max(1.0, float(low_cut))
    high = min(float(high_cut), 0.99 * nyq)

    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return y

    wn = [low / nyq, high / nyq]

    # 設計濾波器
    sos = signal.butter(order, wn, btype="band", output="sos")


    if zero_phase:

        needed_pad = 3 * (2 * sos.shape[0])
        if y.size > needed_pad:
            return signal.sosfiltfilt(sos, y)
        else:
            return signal.sosfilt(sos, y)
    else:
        return signal.sosfilt(sos, y)
    


