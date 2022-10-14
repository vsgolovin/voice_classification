from typing import List, Tuple, Union, Optional, Callable
from pathlib import Path
import random
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio


DATASET_PATH = Path("data")
SUBSET = "dev-clean"


def load_datasets(
        path: Union[Path, str] = "data",
        subset: str = "dev-clean",
        train_size: float = 0.6,
        test_size: float = 0.2,
        seed: Optional[int] = None,
        transform: Optional[Callable] = None):
    path = Path(path)
    df = get_speakers(path / "SPEAKERS.txt")

    if seed is not None:
        random.seed(seed)  # ensures consistent split
    df_train, df_test, df_dev = split_speakers(df, train_size, test_size)

    train_dataset = LibriTTS(path / subset, df_train, transform)
    test_dataset = LibriTTS(path / subset, df_test, transform)
    if df_dev is not None:
        dev_dataset = LibriTTS(path / subset, df_dev, transform)
    else:
        dev_dataset = None

    return train_dataset, test_dataset, dev_dataset


def get_speakers(path: Union[Path, str], subset: str = SUBSET) -> pd.DataFrame:
    """
    Open file `SPEAKERS.txt` and find all speakers from `subset`
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            # skip header lines
            if line[0] == ";":
                continue

            # save all entries that belong to selected subset
            fields = [field.strip() for field in line.split(" | ")]
            assert len(fields) == 5
            if fields[2] == subset:
                fields.pop(2)
                data.append(fields)

    df = pd.DataFrame(data=data, columns=("ID", "SEX", "MINUTES", "NAME"))
    return df


def split_speakers(df: pd.DataFrame, train_size: float = 0.6,
                   test_size: float = 0.2) -> List[pd.DataFrame]:
    """
    Split speakers into train, test and (optionally) dev datasets evenly
    across sexes.
    """
    # convert fractions to sizes
    def get_sizes(total):
        train = int(round(total * train_size))
        test = int(round(total * test_size))
        assert train + test <= total
        dev = total - train - test
        return train, test, dev

    # split dataframe by sex and shuffle
    mask = df["SEX"] == "M"
    m_inds = list(df[mask].index)
    random.shuffle(m_inds)
    f_inds = list(df[~mask].index)
    random.shuffle(f_inds)

    m_sizes = get_sizes(len(m_inds))
    f_sizes = get_sizes(len(f_inds))
    assert not ((m_sizes[-1] == 0) ^ (f_sizes[-1] == 0))

    # create dataframe slices
    dataframes = []
    for i in range(3):
        if m_sizes[i] == 0:
            dataframes.append(None)
        inds = [m_inds.pop() for _ in range(m_sizes[i])]
        inds += [f_inds.pop() for _ in range(f_sizes[i])]
        dataframes.append(df.iloc[inds])

    return dataframes  # train, test, dev


class LibriTTS(Dataset):
    SAMPLE_RATE = 24000

    def __init__(self, path: Union[Path, str], speakers: pd.DataFrame,
                 transform: Optional[Callable] = None):
        self.files = []
        self.labels = []
        for _, speaker in speakers.iterrows():
            cur_path = Path(path) / str(speaker["ID"])
            cur_files = list(cur_path.rglob("*.wav"))
            self.files += cur_files
            self.labels += [speaker["SEX"]] * len(cur_files)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Tensor, int, int]:
        waveform, sample_rate = torchaudio.load(self.files[idx])
        assert sample_rate == self.SAMPLE_RATE,\
            f"File {self.files[idx]}: incorrect sample rate ({sample_rate})"
        label = 0 if self.labels[idx] == "M" else 1
        if self.transform is None:
            return waveform, sample_rate, label
        out = self.transform(waveform)
        return out, label
