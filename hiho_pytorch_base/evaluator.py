"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from .batch import BatchOutput
from .generator import Generator, GeneratorOutput
from .network.predictor import create_padding_mask
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    value: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.value = detach_cpu(self.value)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -1 * output.value


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        output_result: GeneratorOutput = self.generator(
            f0=batch.f0,
            phoneme=batch.phoneme,
            noise_spec=batch.noise_spec,
            speaker_id=batch.speaker_id,
            length=batch.length,
            step_num=self.generator.config.train.diffusion_step_num,
        )

        mask_2d = create_padding_mask(batch.length).squeeze(1)
        pred_spec_all = output_result.spec[mask_2d]
        target_spec_all = batch.target_spec[mask_2d]
        value = l1_loss(pred_spec_all, target_spec_all)

        return EvaluatorOutput(
            value=value,
            data_num=batch.data_num,
        )
