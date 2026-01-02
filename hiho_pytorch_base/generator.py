"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import numpy
import torch
from torch import Tensor, nn

from .config import Config
from .network.predictor import Predictor, create_padding_mask, create_predictor

TensorLike = Tensor | numpy.ndarray


@dataclass
class GeneratorOutput:
    """生成したデータ"""

    spec: Tensor  # (B, L, ?)
    length: Tensor  # (B,)


def to_tensor(array: TensorLike, device: torch.device) -> Tensor:
    """データをTensorに変換する"""
    if not isinstance(array, Tensor | numpy.ndarray):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    tensor = tensor.to(device)
    return tensor


class Generator(nn.Module):
    """生成経路で推論するクラス"""

    def __init__(
        self,
        config: Config,
        predictor: Predictor | Path,
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    @torch.no_grad()
    def forward(
        self,
        *,
        f0: TensorLike,  # (B, L)
        phoneme: TensorLike,  # (B, L)
        noise_spec: TensorLike,  # (B, L, ?)
        speaker_id: TensorLike,  # (B,)
        length: TensorLike,  # (B,)
        step_num: int,
    ) -> GeneratorOutput:
        """生成経路で推論する"""
        f0_t = to_tensor(f0, self.device)
        phoneme_t = to_tensor(phoneme, self.device)
        noise_spec_t = to_tensor(noise_spec, self.device)
        speaker_id_t = to_tensor(speaker_id, self.device)
        length_t = to_tensor(length, self.device).long()

        if self.config.model.flow_type == "rectified_flow":
            spec = self._generate_rectified_flow(
                f0=f0_t,
                phoneme=phoneme_t,
                noise_spec=noise_spec_t,
                speaker_id=speaker_id_t,
                length=length_t,
                step_num=step_num,
            )
        elif self.config.model.flow_type == "meanflow":
            spec = self._generate_meanflow(
                f0=f0_t,
                phoneme=phoneme_t,
                noise_spec=noise_spec_t,
                speaker_id=speaker_id_t,
                length=length_t,
                step_num=step_num,
            )
        else:
            assert_never(self.config.model.flow_type)

        return GeneratorOutput(spec=spec, length=length_t)

    def _generate_rectified_flow(
        self,
        *,
        f0: Tensor,  # (B, L)
        phoneme: Tensor,  # (B, L)
        noise_spec: Tensor,  # (B, L, C)
        speaker_id: Tensor,  # (B,)
        length: Tensor,  # (B,)
        step_num: int,
    ) -> Tensor:
        """RectifiedFlowでスペクトログラムを生成"""
        spec = noise_spec.clone()

        mask = create_padding_mask(length)  # (B, 1, L)
        mask_3d = mask.squeeze(1).unsqueeze(-1)

        t_array = torch.linspace(0, 1, steps=step_num + 1, device=self.device)[:-1]
        delta_t_step = 1.0 / step_num

        for i in range(step_num):
            t = t_array[i].expand(spec.size(0))
            h = torch.zeros_like(t)

            velocity = self.predictor(
                f0=f0,
                phoneme=phoneme,
                input_spec=spec,
                mask=mask,
                t=t,
                h=h,
                speaker_id=speaker_id,
            )

            spec = spec + velocity * delta_t_step * mask_3d

        return spec * mask_3d

    def _generate_meanflow(
        self,
        *,
        f0: Tensor,  # (B, L)
        phoneme: Tensor,  # (B, L)
        noise_spec: Tensor,  # (B, L, C)
        speaker_id: Tensor,  # (B,)
        length: Tensor,  # (B,)
        step_num: int,
    ) -> Tensor:
        """MeanFlowでスペクトログラムを生成"""
        mask = create_padding_mask(length)  # (B, 1, L)
        mask_3d = mask.squeeze(1).unsqueeze(-1)

        if step_num == 1:
            t = torch.ones(noise_spec.size(0), device=self.device)
            h = t
            velocity = self.predictor(
                f0=f0,
                phoneme=phoneme,
                input_spec=noise_spec,
                mask=mask,
                t=t,
                h=h,
                speaker_id=speaker_id,
            )
            return (noise_spec - velocity) * mask_3d

        spec = noise_spec.clone()

        t_array = torch.linspace(1, 0, steps=step_num + 1, device=self.device)
        delta_t_step = 1.0 / step_num

        for i in range(step_num):
            t_start = t_array[i]
            t_end = t_array[i + 1]
            t = t_start.expand(spec.size(0))
            h = (t_start - t_end).expand(spec.size(0))

            velocity = self.predictor(
                f0=f0,
                phoneme=phoneme,
                input_spec=spec,
                mask=mask,
                t=t,
                h=h,
                speaker_id=speaker_id,
            )

            spec = spec - velocity * delta_t_step * mask_3d

        return spec * mask_3d
