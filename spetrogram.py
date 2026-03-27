import sound_detection as sd
from pathlib import Path
import soundfile as sf
import librosa 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram


audio = r"C:\Users\howardhow\Desktop\sound detection\fig_看不出來\看不見\咳一次.wav"
output_path = r"C:\Users\howardhow\Desktop\sound detection\fig_看不出來\看不見\can't see2_bicubic"



y , sr = sf.read(audio)
if y.ndim > 1:
    y = np.mean(y ,axis=1)
y = sd.bandpass_filter(y , sr, low_cut=200, high_cut=2000)
n_fft = 2048

def plot_spectrogram(y ,sr):
    MIN_N = n_fft
    if len(y) < MIN_N:
        y = np.pad(y, (0, MIN_N - len(y)), mode='constant')
    D = librosa.power_to_db(np.abs(librosa.stft(y,n_fft=n_fft)) ** 2 ,ref=np.max)
    return D


def spectrogram(y ,sr , output_path, freq_range = None):
    if len(y) < 2*n_fft:
        f, Pxx = periodogram(y, fs=sr, window='hann', nfft=len(y), scaling='spectrum', detrend=False)
        plt.figure(figsize=(10 , 6))
        Pxx_db = 10 * np.log10(np.maximum(Pxx, 1e-20))

        plt.figure(figsize=(10, 4))
        plt.plot(f, Pxx_db)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.title(f"Periodogram (N={len(y)})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        D = plot_spectrogram(y ,sr)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        if freq_range is not None:
            low, high = freq_range
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            D = D[idx, :]
            freqs = freqs[idx]
        plt.figure(figsize=(10, 6))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

        plt.imshow(D, aspect='auto', origin='lower', cmap='inferno',
                        extent=[0, len(y)/sr, freqs[0], freqs[-1]],
                        vmin=-80, vmax=0, interpolation='bicubic')

        plt.title("Spectrogram")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()


spectrogram(y, sr, output_path, freq_range=(200,2000))

