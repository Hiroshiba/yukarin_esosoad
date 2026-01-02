"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self, assert_never

import torch
from torch import Tensor, nn

from .batch import BatchOutput
from .config import ModelConfig
from .network.predictor import Predictor, create_padding_mask
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    spec_mse_loss: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.spec_mse_loss = detach_cpu(self.spec_mse_loss)
        return self


def masked_mse_loss(  # noqa: D103
    output: Tensor,  # (B, L, C)
    target: Tensor,  # (B, L, C)
    mask_2d: Tensor,  # (B, L)
) -> Tensor:
    """マスク対応のMSE損失を計算"""
    diff_squared = (output - target) ** 2
    masked_loss = diff_squared * mask_2d.unsqueeze(-1)
    denom = mask_2d.sum() * output.size(-1)
    if denom.item() <= 0:
        raise ValueError("mask is empty")
    return masked_loss.sum() / denom


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        if self.model_config.flow_type == "rectified_flow":
            return self._forward_rectified_flow(batch)
        if self.model_config.flow_type == "meanflow":
            return self._forward_meanflow(batch)
        assert_never(self.model_config.flow_type)

    def _forward_rectified_flow(self, batch: BatchOutput) -> ModelOutput:
        """RectifiedFlowの損失を計算"""
        lengths = batch.length  # (B,)
        mask = create_padding_mask(lengths)  # (B, 1, L)
        mask_2d = mask.squeeze(1)  # (B, L)

        h = torch.zeros_like(batch.t)
        velocity = self.predictor(
            f0=batch.f0,
            phoneme=batch.phoneme,
            input_spec=batch.input_spec,
            mask=mask,
            t=batch.t,
            h=h,
            speaker_id=batch.speaker_id,
        )

        target_v = batch.target_spec - batch.noise_spec

        loss = masked_mse_loss(velocity, target_v, mask_2d)

        return ModelOutput(
            loss=loss,
            spec_mse_loss=loss,
            data_num=batch.data_num,
        )

    def _forward_meanflow(self, batch: BatchOutput) -> ModelOutput:
        """MeanFlowの損失を計算"""
        lengths = batch.length  # (B,)
        if (lengths <= 0).any():
            raise ValueError("length must be positive")
        mask = create_padding_mask(lengths)  # (B, 1, L)
        mask_2d = mask.squeeze(1)  # (B, L)

        target_v = batch.noise_spec - batch.target_spec  # (B, L, C)

        def u_func(spec: Tensor, t: Tensor, r: Tensor) -> Tensor:
            """JVP計算用のラッパー関数"""
            h = t - r
            output = self.predictor(
                f0=batch.f0,
                phoneme=batch.phoneme,
                input_spec=spec,
                mask=mask,
                t=t,
                h=h,
                speaker_id=batch.speaker_id,
            )
            return output

        u_pred, du_dt = torch.func.jvp(  # type: ignore[attr-defined]
            func=u_func,
            primals=(batch.input_spec, batch.t, batch.r),
            tangents=(
                target_v,
                torch.ones_like(batch.t),
                torch.zeros_like(batch.r),
            ),
        )

        batch_size = batch.t.shape[0]
        max_length = batch.input_spec.size(1)
        h_expanded = (batch.t - batch.r).unsqueeze(1).expand(batch_size, max_length)

        u_tgt = target_v - h_expanded.unsqueeze(-1) * du_dt

        mse_per_element = (u_pred - u_tgt.detach()) ** 2

        channel = target_v.size(2)
        masked_mse = mse_per_element * mask_2d.unsqueeze(-1)

        denom_all = mask_2d.sum() * channel
        if denom_all.item() <= 0:
            raise ValueError("mask is empty")

        mse = masked_mse.sum() / denom_all

        denom_per_sample = mask_2d.sum(dim=1) * channel
        loss_per_sample = masked_mse.sum(dim=(1, 2)) / denom_per_sample

        adp_wt = (
            loss_per_sample.detach() + self.model_config.adaptive_weighting_eps
        ) ** self.model_config.adaptive_weighting_p
        loss_per_sample = loss_per_sample / adp_wt
        loss = loss_per_sample.mean()

        return ModelOutput(
            loss=loss,
            spec_mse_loss=mse,
            data_num=batch.data_num,
        )
