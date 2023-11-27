import random
from typing import Sequence

import torch
import torchaudio.transforms as T


class SpeedPerturbation:
    def __init__(
        self,
        orig_freq: int,
        factors: Sequence[float],
        probability: float,
    ):
        self.speeder = T.SpeedPerturbation(orig_freq, factors)

        self.orig_freq = orig_freq
        self.probability = probability

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech

        if sample_rate != self.orig_freq:
            speech = T.Resample(sample_rate, self.orig_freq)(speech)

        speech, _ = self.speeder(speech)

        return speech


class TrimAudioSample(object):
    def __init__(
        self,
        factor: float,
        min_length: float,
        max_length: float,
        probability: float,
    ):
        self.factor = factor
        self.min_length = min_length
        self.max_length = max_length
        self.probability = probability

    def apply(self, speech: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() > self.probability:
            return speech

        audio_length = speech.size(1) / sample_rate

        sample_length = self.factor * audio_length
        sample_length = min(self.max_length, sample_length)
        sample_length = max(self.min_length, sample_length)

        max_start_index = (audio_length - sample_length) * sample_rate
        start_index = random.randint(0, max(0, int(max_start_index)))

        length = int(sample_length * sample_rate)
        sample = speech[:, start_index: start_index + length]

        return sample
