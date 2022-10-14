from pathlib import Path
from tqdm import tqdm
import numpy as np
from torchaudio import transforms as T

from utils.dataset import load_datasets, LibriTTS


def main():
    train, test, dev = load_datasets(
        seed=42, transform=T.Spectrogram(n_fft=512))
    save_to = Path("data/arrays")
    print(f"Saving Fourier transforms to {save_to}")
    for dset, name in zip((train, test, dev), ("train", "test", "dev")):
        print(f"{name} dataset:")
        X, y = dataset2arrays(dset)
        np.savetxt(save_to / f"{name}_data.txt", X)
        np.savetxt(save_to / f"{name}_labels.txt", y)


def dataset2arrays(dset: LibriTTS):
    X, y = [], []
    for sp, label in tqdm(dset):
        X.append(sp.squeeze(0).mean(1).numpy())
        y.append(label)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.uint8)


if __name__ == "__main__":
    main()
