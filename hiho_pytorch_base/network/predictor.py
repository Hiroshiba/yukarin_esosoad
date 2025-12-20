"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn

from ..config import NetworkConfig
from .conformer.encoder import Encoder
from .transformer.utility import make_non_pad_mask


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        feature_vector_size: int,
        feature_variable_size: int,
        hidden_size: int,
        target_vector_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        encoder: Encoder,
    ):
        super().__init__()

        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)

        input_size = feature_variable_size + speaker_embedding_size
        self.pre_conformer = nn.Linear(input_size, hidden_size)
        self.encoder = encoder

        self.feature_vector_processor = nn.Linear(feature_vector_size, hidden_size)
        self.vector_head = nn.Linear(hidden_size * 2, target_vector_size)
        self.variable_head = nn.Linear(hidden_size, target_vector_size)
        self.scalar_head = nn.Linear(hidden_size * 2, 1)

    def forward(  # noqa: D102
        self,
        *,
        feature_vector: Tensor,  # (B, ?)
        feature_variable: Tensor,  # (B, L, ?)
        speaker_id: Tensor,  # (B,)
        length: Tensor,  # (B,)
    ) -> tuple[Tensor, Tensor, Tensor]:  # (B, ?), (B, L, ?), (B,)
        device = feature_vector.device
        batch_size = feature_vector.size(0)

        speaker_embedding = self.speaker_embedder(speaker_id)  # (B, ?)

        max_length = feature_variable.size(1)
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(
            batch_size, max_length, -1
        )  # (B, L, ?)

        combined_variable = torch.cat(
            [feature_variable, speaker_expanded], dim=2
        )  # (B, L, ?)

        h = self.pre_conformer(combined_variable)  # (B, L, ?)

        mask = make_non_pad_mask(length).unsqueeze(-2).to(device)  # (B, 1, L)

        encoded, _ = self.encoder(x=h, cond=None, mask=mask)  # (B, L, ?)

        variable_output = self.variable_head(encoded)  # (B, L, ?)

        mask_expanded = mask.squeeze(-2).unsqueeze(-1)  # (B, L, 1)
        masked_encoded = encoded * mask_expanded  # (B, L, ?)
        variable_sum = masked_encoded.sum(dim=1)  # (B, ?)
        variable_mean = variable_sum / length.unsqueeze(-1).float()  # (B, ?)

        fixed_features = self.feature_vector_processor(feature_vector)  # (B, ?)

        final_features = torch.cat([fixed_features, variable_mean], dim=1)  # (B, ?)

        vector_output = self.vector_head(final_features)  # (B, ?)
        scalar_output = self.scalar_head(final_features).squeeze(-1)  # (B,)

        return vector_output, variable_output, scalar_output


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=True,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        feature_vector_size=config.feature_vector_size,
        feature_variable_size=config.feature_variable_size,
        hidden_size=config.hidden_size,
        target_vector_size=config.target_vector_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        encoder=encoder,
    )
