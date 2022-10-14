from typing import Tuple, Callable
from pathlib import Path
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchaudio import transforms as T
from torchvision.transforms import Compose, RandomCrop
from torchvision.models import vgg11, VGG11_Weights
from utils.dataset import load_datasets


BATCH_SIZE = 32
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # load data
    train_dset, _, dev_dset = load_datasets(
        seed=42,
        transform=Compose([
            T.Spectrogram(n_fft=446),
            RandomCrop(224, pad_if_needed=True)
        ])
    )
    train_dataloader = DataLoader(train_dset, BATCH_SIZE, shuffle=True)
    dev_dataloader = DataLoader(dev_dset, BATCH_SIZE, shuffle=True)

    # load and train model
    cnn = get_model().to(DEVICE)
    train_loss, train_acc, val_loss, val_acc = train(
        cnn, train_dataloader, dev_dataloader, epochs=EPOCHS)

    epochs = np.arange(0, EPOCHS) + 1
    plt.plot(epochs, train_loss, 'b-', label="train")
    plt.plot(epochs, val_loss, 'r-', label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.twinx()
    plt.plot(epochs, train_acc * 100, 'b:', label="train")
    plt.plot(epochs, val_acc * 100, 'r:', label="validation")
    plt.ylabel("Accuracy, %")
    plt.ylim(48, 102)
    plt.legend()
    plt.savefig(Path("plots/cnn_model.png"), dpi=150)


def get_transform():
    return Compose([
        T.Spectrogram(n_fft=446),
        RandomCrop(224, pad_if_needed=True)
    ])


def get_model():
    cnn = vgg11(weights=VGG11_Weights.DEFAULT)
    cnn.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1)
    )
    return cnn


def train(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
          epochs: int = 5, save_weights: bool = True) -> Tuple[np.ndarray]:
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)
    train_loss = np.zeros(epochs)
    train_acc = np.zeros_like(train_loss)
    val_loss = np.ones_like(train_loss) * np.inf
    val_acc = np.zeros_like(train_loss)

    for epoch in trange(epochs):
        model.train()
        total_loss = 0.
        hits = 0     # accurate predictions
        samples = 0  # total input signals (== len(dataset))

        # train one epoch
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X.expand(-1, 3, -1, -1)).squeeze(1)
            loss = loss_fn(out, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X)
            hits += int((torch.sigmoid(out).round().int() == y).sum())
            samples += len(X)

        # save metrics
        train_loss[epoch] = total_loss / samples
        train_acc[epoch] = hits / samples

        # evaluate at dev (validation) subset
        val_loss[epoch], val_acc[epoch] = evaluate(model, val_dl,
                                                   loss_fn=loss_fn)
        if save_weights and np.argmin(val_loss) == epoch:
            torch.save(model.state_dict(), Path("models/cnn.pth"))

    return train_loss, train_acc, val_loss, val_acc


@torch.no_grad()
def evaluate(model: nn.Module, val_dataloader: DataLoader, loss_fn: Callable):
    total_loss = 0.0
    hits = 0
    samples = 0
    model.eval()
    for X, y in val_dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X.expand(-1, 3, -1, -1)).squeeze(1)
        total_loss += loss_fn(out, y.float()).item() * len(X)
        hits += int((torch.sigmoid(out).round().int() == y).sum())
        samples += len(X)
    return total_loss / samples, hits / samples


if __name__ == "__main__":
    main()
