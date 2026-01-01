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
    spec: Tensor  # (B, L, ?)
    speaker_id: Tensor  # (B,)
    length: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.speaker_id.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.f0 = to_device(self.f0, device, non_blocking=non_blocking)
        self.phoneme = to_device(self.phoneme, device, non_blocking=non_blocking)
        self.spec = to_device(self.spec, device, non_blocking=non_blocking)
        self.speaker_id = to_device(self.speaker_id, device, non_blocking=non_blocking)
        self.length = to_device(self.length, device, non_blocking=non_blocking)
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
        spec=pad_sequence([d.spec for d in data_list], batch_first=True),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
        length=torch.tensor([d.f0.shape[0] for d in data_list]),
    )
