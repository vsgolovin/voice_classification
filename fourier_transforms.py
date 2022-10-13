from typing import Tuple
from pathlib import Path
import torch
from torchaudio import transforms as T
import numpy as np
from scipy.fft import fft
from dataset import load_datasets


def scipy_dft(waveform: torch.Tensor, sample_rate: int
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute discrete Fourier transform of real-valued `waveform`.
    Returns DFT amplitude spectrum (frequency, abs(DFT)).
    """
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    N = len(waveform)
    dft = np.abs(fft(np.asarray(waveform)))
    freq = sample_rate * np.arange(N // 2 + 1) / N
    return freq, dft[:N // 2 + 1] / (N**0.5)


def torch_averaged_stft(waveform, sr, n_fft=512
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute STFT of `waveform` and average it over time.
    Returns averaged STFT amplitude spectrum (frequency, abs(STFT_avg)).
    """
    spec_f = T.Spectrogram(n_fft=n_fft, power=1)
    stft = spec_f(waveform)
    if stft.ndim == 3:
        stft = stft.squeeze(0)
    stft = torch.mean(stft, dim=1)
    freq = sr * np.arange(n_fft // 2 + 1) / n_fft
    return freq, np.array(stft.squeeze())


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rc("lines", linewidth=0.8)

    # load data
    df_train, df_test, _ = load_datasets(seed=42)
    w, sr, _ = df_train[0]

    # compute DFT with scipy
    freq, dft = scipy_dft(w, sr)
    dft /= dft.max()

    plt.figure()
    plt.plot(freq, dft, label="scipy")

    # compute averaged STFT with torchaudio
    freq, stft = torch_averaged_stft(w, sr, n_fft=4096)
    stft /= stft.max()
    plt.plot(freq, stft, label="torchaudio")

    plt.xlabel("Frequency (Hz)")
    plt.title("Amplitude spectrum")
    plt.legend()

    plt.savefig(Path("plots") / "scipy_torch_comparison.png", dpi=150)
