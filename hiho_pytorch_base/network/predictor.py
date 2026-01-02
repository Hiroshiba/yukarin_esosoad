"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn

from ..config import NetworkConfig
from ..data.statistics import DataStatistics
from .conformer.encoder import Encoder
from .transformer.utility import make_non_pad_mask


def create_padding_mask(
    lengths: Tensor,  # (B,)
) -> Tensor:  # (B, 1, L)
    """lengthsからパディングマスクを生成"""
    mask = make_non_pad_mask(lengths).unsqueeze(-2).to(lengths.device)
    return mask


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        f0_embedding_size: int,
        hidden_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        output_size: int,
        encoder: Encoder,
        statistics: DataStatistics,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        if len(statistics.spec_mean) != speaker_size:
            raise ValueError(
                f"statistics speaker_size mismatch: network={speaker_size} statistics={len(statistics.spec_mean)}"
            )

        self.register_buffer("spec_mean", torch.from_numpy(statistics.spec_mean))
        self.register_buffer("spec_std", torch.from_numpy(statistics.spec_std))

        self.phoneme_embedder = nn.Sequential(
            nn.Embedding(phoneme_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
        )

        self.speaker_embedder = nn.Sequential(
            nn.Embedding(speaker_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
        )
        self.f0_linear = nn.Linear(1, f0_embedding_size)

        embedding_size = (
            f0_embedding_size
            + phoneme_embedding_size
            + speaker_embedding_size
            + output_size
            + 1
            + 1
        )
        self.pre = nn.Linear(embedding_size, hidden_size)

        self.encoder = encoder

        self.post = nn.Linear(hidden_size, output_size)

    def forward(  # noqa: D102
        self,
        *,
        f0: Tensor,  # (B, L)
        phoneme: Tensor,  # (B, L)
        input_spec: Tensor,  # (B, L, ?)
        mask: Tensor,  # (B, 1, L)
        t: Tensor,  # (B,)
        h: Tensor,  # (B,)
        speaker_id: Tensor,  # (B,)
    ) -> Tensor:  # (B, L, ?)
        device = speaker_id.device
        batch_size = speaker_id.shape[0]
        max_length = phoneme.size(1)

        h_phoneme = self.phoneme_embedder(phoneme)  # (B, L, ?)
        h_f0 = self.f0_linear(f0.unsqueeze(-1))  # (B, L, ?)

        h_speaker = self.speaker_embedder(speaker_id)  # (B, ?)
        h_speaker = h_speaker.unsqueeze(1).expand(batch_size, max_length, -1)

        t_expanded = t.unsqueeze(1).unsqueeze(2).expand(batch_size, max_length, 1)
        h_expanded = h.unsqueeze(1).unsqueeze(2).expand(batch_size, max_length, 1)

        x = torch.cat(
            [h_f0, h_phoneme, h_speaker, input_spec, t_expanded, h_expanded],
            dim=2,
        )
        x = self.pre(x)

        x, _ = self.encoder(x=x, cond=None, mask=mask.to(device))
        y = self.post(x)
        return y


def create_predictor(config: NetworkConfig, *, statistics: DataStatistics) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=False,
        use_conv_glu_module=False,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        f0_embedding_size=config.f0_embedding_size,
        hidden_size=config.hidden_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        output_size=config.output_size,
        encoder=encoder,
        statistics=statistics,
    )
