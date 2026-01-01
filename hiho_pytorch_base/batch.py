"""バッチ処理モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .data.data import OutputData
from .utility.pytorch_utility import to_device


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    f0: Tensor  # (B, L)
    phoneme: Tensor  # (B, L)
    input_spec: Tensor  # (B, L, ?)
    target_spec: Tensor  # (B, L, ?)
    noise_spec: Tensor  # (B, L, ?)
    speaker_id: Tensor  # (B,)
    length: Tensor  # (B,)
    t: Tensor  # (B,)
    r: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.speaker_id.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.f0 = to_device(self.f0, device, non_blocking=non_blocking)
        self.phoneme = to_device(self.phoneme, device, non_blocking=non_blocking)
        self.input_spec = to_device(self.input_spec, device, non_blocking=non_blocking)
        self.target_spec = to_device(
            self.target_spec, device, non_blocking=non_blocking
        )
        self.noise_spec = to_device(self.noise_spec, device, non_blocking=non_blocking)
        self.speaker_id = to_device(self.speaker_id, device, non_blocking=non_blocking)
        self.length = to_device(self.length, device, non_blocking=non_blocking)
        self.t = to_device(self.t, device, non_blocking=non_blocking)
        self.r = to_device(self.r, device, non_blocking=non_blocking)
        return self


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        f0=pad_sequence([d.f0 for d in data_list], batch_first=True),
        phoneme=pad_sequence([d.phoneme for d in data_list], batch_first=True),
        input_spec=pad_sequence([d.input_spec for d in data_list], batch_first=True),
        target_spec=pad_sequence([d.target_spec for d in data_list], batch_first=True),
        noise_spec=pad_sequence([d.noise_spec for d in data_list], batch_first=True),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
        length=torch.tensor([d.f0.shape[0] for d in data_list]),
        t=collate_stack([d.t for d in data_list]),
        r=collate_stack([d.r for d in data_list]),
    )
