from typing import Tuple, List, Any, Optional

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn.functional as F

import librosa
import numpy as np


def extract_cqt_spectrum(
    signal: torch.Tensor,
    sr: int,
    hop_length: Optional[int] = 512,
    n_bins: Optional[int] = 84,
    bins_per_octave: Optional[int] = 12,
    average_factor: Optional[int] = 20,
):
    signal = signal.mean(dim=0, keepdim=True)
    signal = signal.squeeze(0).numpy()

    C = np.abs(
        librosa.cqt(
            signal,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )
    C = librosa.amplitude_to_db(C, ref=np.max)

    C = torch.from_numpy(C).unsqueeze(0)
    C = F.avg_pool1d(C, average_factor)[0]

    return C


def build_augmentation(config: DictConfig) -> Tuple[List[Any], List[Any]]:
    augment_config = config.get("audio_augment", {})
    audio_augments = [instantiate(cfg) for cfg in augment_config.values()]

    augment_config = config.get("feature_augment", {})
    feature_augments = [instantiate(cfg) for cfg in augment_config.values()]

    return audio_augments, feature_augments
