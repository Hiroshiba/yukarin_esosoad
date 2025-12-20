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

    feature_vector: Tensor  # (B, ?)
    feature_variable: Tensor  # (B, L, ?)
    target_vector: Tensor  # (B, ?)
    target_variable: Tensor  # (B, L, ?)
    target_scalar: Tensor  # (B,)
    speaker_id: Tensor  # (B,)
    length: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.feature_vector.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.feature_vector = to_device(
            self.feature_vector, device, non_blocking=non_blocking
        )
        self.feature_variable = to_device(
            self.feature_variable, device, non_blocking=non_blocking
        )
        self.target_vector = to_device(
            self.target_vector, device, non_blocking=non_blocking
        )
        self.target_variable = to_device(
            self.target_variable, device, non_blocking=non_blocking
        )
        self.target_scalar = to_device(
            self.target_scalar, device, non_blocking=non_blocking
        )
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
        feature_vector=collate_stack([d.feature_vector for d in data_list]),
        feature_variable=pad_sequence(
            [d.feature_variable for d in data_list], batch_first=True
        ),
        target_vector=collate_stack([d.target_vector for d in data_list]),
        target_variable=pad_sequence(
            [d.target_variable for d in data_list], batch_first=True
        ),
        target_scalar=collate_stack([d.target_scalar for d in data_list]),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
        length=torch.tensor([d.feature_variable.shape[0] for d in data_list]),
    )
