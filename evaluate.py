from pathlib import Path
import pickle
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits

from utils.dataset import load_datasets
from ft_export import get_array_paths, N_FFT
from train_fc import SimpleNeuralNetwork, evaluate as evaluate_fc
from train_cnn import get_model, get_transform
from train_cnn import evaluate as evaluate_cnn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Scikit-learn models:")
    test_sklearn_models([
        Path("models/dtree.pkl"),
        Path("models/svm.pkl")
    ])
    print()

    test_fc(Path("models/linear.pth"))
    test_cnn(Path("models/cnn.pth"))


def test_sklearn_models(paths: Iterable[Path]):
    "Load pickled `sklearn` models check their performance at test set"
    X, y = (np.loadtxt(f) for f in get_array_paths("test"))
    for path in paths:
        if not path.exists():
            print(f"Error: could not load {path}")
            continue
        print(f"Loading from {path}")
        with open(path, "rb") as infile:
            clf = pickle.load(infile)
        pred = clf.predict(X)
        acc = np.sum(pred == y) / len(y)
        print(f"Model accuracy: {acc*100:.1f}%")


def test_fc(path: Path):
    X, y = (np.loadtxt(f) for f in get_array_paths("test"))
    model = SimpleNeuralNetwork(input_dim=N_FFT // 2 + 1).to(DEVICE)
    model.load_state_dict(torch.load(path))
    _, acc = evaluate_fc(model, X, y, loss_fn=binary_cross_entropy_with_logits,
                         bs=64, device=DEVICE)
    print(f"Fully-connected neural network accuracy: {acc*100:.1f}%")


def test_cnn(path: Path):
    _, test_dset, _ = load_datasets(
        seed=42,
        transform=get_transform()
    )
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(path))
    test_dl = DataLoader(test_dset, batch_size=32)
    _, acc = evaluate_cnn(model, test_dl,
                          loss_fn=binary_cross_entropy_with_logits,
                          device=DEVICE)
    print(f"Convolutional neural network test accuracy: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
