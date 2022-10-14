import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.fourier_transforms import scipy_dft
from utils.dataset import load_datasets


def main():
    # load data
    train, _, dev = load_datasets(seed=42)

    # calculate average frequencies of all waveforms
    freq_male = []
    freq_female = []
    for dset in (train, dev):
        for wfm, sr, label in tqdm(dset):
            avg_freq = get_average_freq(wfm, sr)
            if label == 0:
                freq_male.append(avg_freq)
            else:
                assert label == 1
                freq_female.append(avg_freq)

    # plot results
    fig = plt.figure()
    ax = fig.gca()
    ax.boxplot([freq_male, freq_female])
    ax.set_xticklabels(["male", "female"])
    ax.set_ylim(0, 200)
    ax.set_title("Average frequencies")
    fig.savefig("plots/avg_frequencies.png", dpi=150)


def get_average_freq(waveform: torch.Tensor, sample_rate: int) -> float:
    freq, dft = scipy_dft(waveform, sample_rate)  # amplitude
    avg_freq = np.mean(freq * dft)
    return avg_freq


if __name__ == "__main__":
    main()
