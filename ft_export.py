from typing import Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torchaudio import transforms as T

from utils.dataset import load_datasets, LibriTTS


ARRAY_PATH = Path("data/arrays")
N_FFT = 512


def main():
    train, test, dev = load_datasets(
        seed=42, transform=T.Spectrogram(n_fft=N_FFT))
    save_to = ARRAY_PATH
    print(f"Saving Fourier transforms to {save_to}")
    for dset, name in zip((train, test, dev), ("train", "test", "dev")):
        print(f"{name} dataset:")
        X, y = dataset2arrays(dset)
        path_X, path_y = get_array_paths(name)
        np.savetxt(path_X, X)
        np.savetxt(path_y, y)


def dataset2arrays(dset: LibriTTS):
    X, y = [], []
    for sp, label in tqdm(dset):
        X.append(sp.squeeze(0).mean(1).numpy())
        y.append(label)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.uint8)


def get_array_paths(subset: str = "train") -> Tuple[Path, Path]:
    "Get paths to data and labels from selected database subset"
    assert subset in ("train", "test", "dev"), f"Unknown split {subset}"
    return (ARRAY_PATH / f"{subset}_data.txt",
            ARRAY_PATH / f"{subset}_labels.txt")


if __name__ == "__main__":
    main()
