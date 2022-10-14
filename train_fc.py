from typing import Callable, Tuple
from pathlib import Path
from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from ft_export import get_array_paths, N_FFT


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 20


def main():
    X_train, y_train = (np.loadtxt(fname)
                        for fname in get_array_paths("train"))
    X_dev, y_dev = (np.loadtxt(fname) for fname in get_array_paths("dev"))

    model = SimpleNeuralNetwork(input_dim=N_FFT // 2 + 1).to(DEVICE)

    train_loss, train_acc, val_loss, val_acc = train(
        model, X_train, y_train, X_dev, y_dev)

    plt.figure()
    epochs = np.arange(len(train_loss)) + 1
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
    plt.savefig(Path("plots/fc_model.png"), dpi=150)


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
          X_dev: np.ndarray, y_dev: np.ndarray, bs: int = BATCH_SIZE,
          epochs: int = EPOCHS, save_weights: bool = True
          ) -> Tuple[np.ndarray]:
    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_losses = np.zeros(epochs)
    train_acc = np.zeros_like(train_losses)
    val_losses = np.ones_like(train_losses) * np.inf  # for finding minimum
    val_acc = np.zeros_like(train_losses)

    for epoch in trange(epochs):
        model.train()
        inds = np.random.permutation(np.arange(len(X_train)))
        total_loss = 0.0
        correct = 0

        for iter_num in range(len(X_train) // bs):
            # select batch
            ix = inds[iter_num*bs:(iter_num+1)*bs]
            X = torch.tensor(X_train[ix], device=DEVICE, dtype=torch.float32)
            y = torch.tensor(y_train[ix], device=DEVICE, dtype=torch.float32)

            # compute predictions and update model
            output = model(X).squeeze(1)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate metrics
            total_loss += loss.item() * len(X)
            predictions = np.round(
                torch.sigmoid(output.detach()).cpu().numpy()
            ).astype("int")
            correct += (predictions == y_train[ix]).sum()

        train_losses[epoch] = total_loss / len(X_train)
        train_acc[epoch] = correct / len(X_train)
        val_losses[epoch], val_acc[epoch] = evaluate(
            model, X_dev, y_dev, loss_fn, bs)
        if save_weights and np.argmin(val_losses) == epoch:
            torch.save(model.state_dict(), Path("models/linear.pth"))

    return train_losses, train_acc, val_losses, val_acc


@torch.no_grad()
def evaluate(model: nn.Module, X_dev: np.ndarray, y_dev: np.ndarray,
             loss_fn: Callable, bs: int = BATCH_SIZE) -> Tuple[float, float]:
    total_loss = 0.0
    correct_predictions = 0
    model.eval()
    inds = np.random.permutation(np.arange(len(X_dev)))

    for iter_num in range(len(X_dev) // bs):
        ix = inds[iter_num*bs:(iter_num+1)*bs]
        X, y = (torch.tensor(arr, device=DEVICE, dtype=torch.float32)
                for arr in (X_dev[ix], y_dev[ix]))

        output = model(X).squeeze(1)
        total_loss += loss_fn(output, y).item() * bs
        predictions = np.round(torch.sigmoid(output.detach()).cpu().numpy()
                               ).astype("int")
        correct_predictions += (predictions == y_dev[ix]).sum()

    return total_loss / len(X_dev), correct_predictions / len(X_dev)


if __name__ == "__main__":
    main()
