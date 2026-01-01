"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor

from .phoneme import ArpaPhoneme
from .sampling_data import (
    ResampleInterpolateKind,
    SamplingData,
)


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    phonemes: list[ArpaPhoneme]
    f0_data: SamplingData
    volume_data: SamplingData
    silence_data: SamplingData
    spec_data: SamplingData  # NOTE: 対数メルスペクトログラム
    speaker_id: int


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    f0: Tensor  # (L,)
    phoneme: Tensor  # (L,)
    spec: Tensor  # (L, ?) NOTE: 対数メルスペクトログラム
    speaker_id: Tensor


def get_notsilence_range(silence: numpy.ndarray, prepost_silence_length: int):
    """
    最初と最後の無音を除去したrangeを返す。

    一番最初や最後が無音でない場合はノイズとみなしてその区間も除去する。
    最小でもprepost_silence_lengthだけは確保する。
    """
    length = len(silence)

    ps = numpy.argwhere(numpy.logical_and(silence[:-1], ~silence[1:]))
    pre_length = ps[0][0] + 1 if len(ps) > 0 else 0
    pre_index = max(0, pre_length - prepost_silence_length)

    ps = numpy.argwhere(numpy.logical_and(~silence[:-1], silence[1:]))
    post_length = length - (ps[-1][0] + 1) if len(ps) > 0 else 0
    post_index = length - max(0, post_length - prepost_silence_length)
    return range(pre_index, post_index)


def create_frame_vowel_f0s(
    f0: numpy.ndarray,
    volume: numpy.ndarray,
    vowel_index: numpy.ndarray,
    durations: numpy.ndarray,
    frame_rate: float,
) -> numpy.ndarray:
    """母音ごとの重み付け平均を計算し、フレームにブロードキャストする"""
    if len(vowel_index) == 0:
        raise ValueError(
            "母音インデックスが空です。LABファイルに母音が含まれていない可能性があります。"
        )

    # 音素の時間範囲を計算
    phoneme_times = numpy.cumsum(numpy.concatenate([[0], durations]))
    vowel_start_times = phoneme_times[vowel_index]
    vowel_end_times = phoneme_times[vowel_index + 1]
    vowel_start_frames = (vowel_start_times * frame_rate).astype(int)
    vowel_end_frames = (vowel_end_times * frame_rate).astype(int)

    # F0をNaNに変換（F0=0は無声区間）
    f0_masked = f0.copy().astype(float)
    f0_masked[f0_masked == 0] = numpy.nan

    # dB → 振幅変換
    volume_amplitude = numpy.power(10, volume / 20.0)

    # 各母音セグメントを処理
    frame_length = int(phoneme_times[-1] * frame_rate)
    output_f0 = numpy.zeros(frame_length, dtype=numpy.float32)

    for start_frame, end_frame in zip(
        vowel_start_frames, vowel_end_frames, strict=True
    ):
        f0_segment = f0_masked[start_frame:end_frame]
        volume_segment = volume_amplitude[start_frame:end_frame]

        # 有効なF0値のみで重み付け平均を計算
        valid_mask = ~numpy.isnan(f0_segment)

        if numpy.any(valid_mask) and numpy.sum(volume_segment[valid_mask]) > 0:
            weighted_mean = numpy.sum(
                f0_segment[valid_mask] * volume_segment[valid_mask]
            ) / numpy.sum(volume_segment[valid_mask])
            value = weighted_mean
        else:
            value = 0.0
        output_f0[start_frame:end_frame] = value

    return output_f0


def create_frame_phoneme_ids(
    phonemes: list[ArpaPhoneme], frame_rate: float
) -> numpy.ndarray:
    """音素継続時間からフレームごとの音素ID配列を生成する"""
    frame_length = int(phonemes[-1].end * frame_rate)
    phoneme_ids = numpy.zeros(frame_length, dtype=numpy.int64)

    for phoneme in phonemes:
        start_frame = int(phoneme.start * frame_rate)
        end_frame = int(phoneme.end * frame_rate)
        end_frame = min(end_frame, frame_length)
        pid = ArpaPhoneme.phoneme_list.index(phoneme.phoneme)
        phoneme_ids[start_frame:end_frame] = pid

    return phoneme_ids


def preprocess(
    d: InputData,
    prepost_silence_length: int,
    max_sampling_length: int,
    is_eval: bool,
) -> OutputData:
    """全ての変換・検証・配列化処理を統合"""
    # リサンプリング
    frame_rate = float(d.spec_data.rate)
    f0 = d.f0_data.resample(
        sampling_rate=frame_rate, index=0, kind=ResampleInterpolateKind.nearest
    )
    volume = d.volume_data.resample(
        sampling_rate=frame_rate, index=0, kind=ResampleInterpolateKind.nearest
    )
    silence = d.silence_data.resample(
        sampling_rate=frame_rate, index=0, kind=ResampleInterpolateKind.nearest
    )
    spec = d.spec_data.array

    phoneme_id = create_frame_phoneme_ids(d.phonemes, frame_rate=frame_rate)

    # 母音ごとのF0重み付け平均を計算
    phoneme_durations = numpy.array(
        [p.duration for p in d.phonemes], dtype=numpy.float32
    )
    vowel_indices = [
        i
        for i, phoneme in enumerate(d.phonemes)
        if ArpaPhoneme.is_vowel(phoneme.phoneme)
    ]
    f0 = create_frame_vowel_f0s(
        f0=f0,
        volume=volume,
        vowel_index=numpy.array(vowel_indices),
        durations=phoneme_durations,
        frame_rate=frame_rate,
    )

    # 長さと一貫性の検証（許容誤差は3フレーム）
    len_spec = int(len(spec))
    len_f0 = int(len(f0))
    len_vol = int(len(volume))
    len_lab = int(len(phoneme_id))
    len_sil = int(len(silence))

    def _check_pair(a: int, b: int, what: str) -> None:
        if abs(a - b) > 3:
            raise ValueError(
                f"{what} の長さが一致しません: {a} vs {b} (許容:3フレーム)"
            )

    _check_pair(len_spec, len_f0, "spec と f0")
    _check_pair(len_spec, len_vol, "spec と volume")
    _check_pair(len_spec, len_lab, "spec と LAB 由来フレーム数")
    _check_pair(len_spec, len_sil, "spec と silence")

    # 長さを統一
    frame_length = min(len_spec, len_f0, len_vol, len_lab, len_sil)
    spec = spec[:frame_length]
    f0 = f0[:frame_length]
    volume = volume[:frame_length]
    silence = silence[:frame_length]
    phoneme_id = phoneme_id[:frame_length]

    # 最初と最後の無音を除去
    notsilence_range = get_notsilence_range(
        silence=silence, prepost_silence_length=prepost_silence_length
    )
    f0 = f0[notsilence_range]
    phoneme_id = phoneme_id[notsilence_range]
    spec = spec[notsilence_range]

    # 最大サンプリング長
    length = len(f0)
    if length > max_sampling_length:
        if is_eval:
            offset = 0
        else:
            offset = numpy.random.default_rng().integers(
                length - max_sampling_length + 1
            )
        s = slice(offset, offset + max_sampling_length)
        f0 = f0[s]
        phoneme_id = phoneme_id[s]
        spec = spec[s]

    # Tensor変換
    return OutputData(
        f0=torch.from_numpy(f0).float(),
        phoneme=torch.from_numpy(phoneme_id).long(),
        spec=torch.from_numpy(spec).float(),
        speaker_id=torch.tensor(d.speaker_id).long(),
    )
