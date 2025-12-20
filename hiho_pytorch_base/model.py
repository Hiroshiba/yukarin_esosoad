"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, mse_loss

from .batch import BatchOutput
from .config import ModelConfig
from .network.predictor import Predictor
from .network.transformer.utility import make_non_pad_mask
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    loss_vector: Tensor
    loss_variable: Tensor
    loss_scalar: Tensor
    accuracy: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.loss_vector = detach_cpu(self.loss_vector)
        self.loss_variable = detach_cpu(self.loss_variable)
        self.loss_scalar = detach_cpu(self.loss_scalar)
        self.accuracy = detach_cpu(self.accuracy)
        return self


def accuracy(
    output: Tensor,  # (B, ?)
    target: Tensor,  # (B,)
) -> Tensor:
    """分類精度を計算"""
    with torch.no_grad():
        indexes = torch.argmax(output, dim=1)  # (B,)
        correct = torch.eq(indexes, target).view(-1)  # (B,)
        return correct.float().mean()


def masked_mse_loss(
    output: Tensor,  # (B, L, ?)
    target: Tensor,  # (B, L, ?)
    mask: Tensor,  # (B, L)
) -> Tensor:
    """マスク対応のMSE損失を計算"""
    diff_squared = (output - target) ** 2  # (B, L, ?)
    mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
    masked_loss = diff_squared * mask_expanded  # (B, L, ?)
    return masked_loss.sum() / (mask.sum() * output.size(-1))


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        (
            vector_output,  # (B, ?)
            variable_output,  # (B, L, ?)
            scalar_output,  # (B,)
        ) = self.predictor(
            feature_vector=batch.feature_vector,
            feature_variable=batch.feature_variable,
            speaker_id=batch.speaker_id,
            length=batch.length,
        )

        target_vector = batch.target_vector  # (B,)
        target_variable = batch.target_variable  # (B, L, ?)
        target_scalar = batch.target_scalar  # (B,)

        mask = make_non_pad_mask(batch.length)  # (B, L)

        loss_vector = cross_entropy(vector_output, target_vector)
        loss_variable = masked_mse_loss(variable_output, target_variable, mask)
        loss_scalar = mse_loss(scalar_output, target_scalar)
        total_loss = loss_vector + loss_variable + loss_scalar
        acc = accuracy(vector_output, target_vector)

        return ModelOutput(
            loss=total_loss,
            loss_vector=loss_vector,
            loss_variable=loss_variable,
            loss_scalar=loss_scalar,
            accuracy=acc,
            data_num=batch.data_num,
        )
