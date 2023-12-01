import os
import random
from typing import Tuple, List, Union, Any, Optional

from omegaconf import DictConfig

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from lightcover.datas.audio import extract_cqt_spectrum, build_augmentation
from lightcover.utils.common import build_dataset


def collate_csi_data(batch: List[Any]) -> Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    lengths = [len(feat) for feat in features]
    features = pad_sequence(features, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)

    targets = [b[1] for b in batch]
    targets = torch.tensor(targets, dtype=torch.long)

    return features, lengths, targets


class AudioDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.dataset = build_dataset(filepaths)
        self.audio_augment, self.feature_augment = [], []
        if augmentation is not None:
            augments = build_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augments

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        data = self.dataset[index]
        filepath = data["filepath"]
        label = int(data["id"])

        signal, sr = torchaudio.load(filepath)
        for augment in self.audio_augment:
            signal = augment.apply(signal, sr)

        spectrum = extract_cqt_spectrum(signal, sr)
        for augment in self.feature_augment:
            spectrum = augment.apply(spectrum)

        return spectrum.T, label

    def __len__(self) -> int:
        return len(self.dataset)


class TensorDataset(Dataset):
    def __init__(
        self,
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.dataset = build_dataset(filepaths)
        self.audio_augment, self.feature_augment = [], []
        if augmentation is not None:
            augments = build_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augments

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        data = self.dataset[index]
        filepath = data["filepath"]
        label = data["id"]

        dirname = os.path.dirname(filepath).replace("/wav", "/cqt")
        filename = os.path.basename(filepath).replace(".wav", ".pt")

        speed = random.choice([0.8, 0.9, 1.0, 1.1, 1.2])
        filepath = os.path.join(dirname, f"{speed}___{filename}")

        spectrum = torch.load(filepath, map_location="cpu")
        for augment in self.feature_augment:
            spectrum = augment.apply(spectrum)

        return spectrum.T, label

    def __len__(self) -> int:
        return len(self.dataset)
