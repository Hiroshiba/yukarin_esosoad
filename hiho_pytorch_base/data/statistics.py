"""統計情報モジュール"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Self

import numpy
from upath import UPath

from ..config import DataFileConfig, DatasetConfig
from ..data.sampling_data import SamplingData
from ..utility.upath_utility import to_local_path


@dataclass
class DataStatistics:
    """話者ごとの統計情報"""

    spec_mean: numpy.ndarray
    spec_std: numpy.ndarray

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """辞書から統計情報を生成"""
        return cls(
            spec_mean=numpy.asarray(d["spec_mean"], dtype=numpy.float64),
            spec_std=numpy.asarray(d["spec_std"], dtype=numpy.float64),
        )

    def to_dict(self) -> dict[str, Any]:
        """統計情報を辞書に変換"""
        return {
            "spec_mean": self.spec_mean.tolist(),
            "spec_std": self.spec_std.tolist(),
        }


def _get_statistics_cache_key_and_info(
    config: DataFileConfig,
) -> tuple[str, dict[str, str | None]]:
    root_dir = None if config.root_dir is None else str(config.root_dir)

    speaker_dict_text = to_local_path(config.speaker_dict_path).read_text()
    speaker_dict_hash = hashlib.sha256(
        speaker_dict_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    spec_pathlist_text = to_local_path(config.spec_pathlist_path).read_text()
    spec_pathlist_hash = hashlib.sha256(
        spec_pathlist_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    info = {
        "root_dir": root_dir,
        "spec_pathlist_path": str(config.spec_pathlist_path),
        "spec_pathlist_hash": spec_pathlist_hash,
        "speaker_dict_path": str(config.speaker_dict_path),
        "speaker_dict_hash": speaker_dict_hash,
    }

    cache_key = hashlib.sha256(
        json.dumps(info, sort_keys=True, ensure_ascii=False).encode(
            "utf-8", errors="surrogatepass"
        )
    ).hexdigest()
    return cache_key, info


@dataclass(frozen=True)
class StatisticsDataInput:
    """統計情報計算用データ"""

    spec_path: UPath
    speaker_id: int


def _load_statistics_item(
    d: StatisticsDataInput,
) -> tuple[int, int, float, float]:
    spec_data = SamplingData.load(to_local_path(d.spec_path))
    spec = spec_data.array.astype(numpy.float64)

    spec_count = int(spec.size)
    spec_sum = float(spec.sum())
    spec_sumsq = float((spec * spec).sum())

    return (
        d.speaker_id,
        spec_count,
        spec_sum,
        spec_sumsq,
    )


def _calc_statistics(
    datas: list[StatisticsDataInput],
    *,
    workers: int,
) -> DataStatistics:
    """話者ごとの統計情報を取得"""
    if workers <= 0:
        raise ValueError(f"workers must be > 0: {workers}")
    if len(datas) == 0:
        raise ValueError("datas is empty")

    max_speaker_id = max(d.speaker_id for d in datas)
    speaker_size = max_speaker_id + 1

    spec_count = numpy.zeros(speaker_size, dtype=numpy.int64)
    spec_sum = numpy.zeros(speaker_size, dtype=numpy.float64)
    spec_sumsq = numpy.zeros(speaker_size, dtype=numpy.float64)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_load_statistics_item, d) for d in datas]
        for future in as_completed(futures):
            (
                speaker_id,
                item_spec_count,
                item_spec_sum,
                item_spec_sumsq,
            ) = future.result()

            spec_count[speaker_id] += item_spec_count
            spec_sum[speaker_id] += item_spec_sum
            spec_sumsq[speaker_id] += item_spec_sumsq

    spec_mean = numpy.full(speaker_size, numpy.nan, dtype=numpy.float64)
    spec_std = numpy.full(speaker_size, numpy.nan, dtype=numpy.float64)

    for speaker_id in range(speaker_size):
        if spec_count[speaker_id] > 0:
            mean = spec_sum[speaker_id] / spec_count[speaker_id]
            var = spec_sumsq[speaker_id] / spec_count[speaker_id] - mean * mean
            spec_mean[speaker_id] = mean
            spec_std[speaker_id] = numpy.sqrt(var)

    return DataStatistics(spec_mean=spec_mean, spec_std=spec_std)


def get_or_calc_statistics(
    config: DatasetConfig,
    datas: list[StatisticsDataInput],
    *,
    workers: int,
) -> DataStatistics:
    """統計情報を取得または計算する"""
    cache_key, info = _get_statistics_cache_key_and_info(config.train)
    cache_dir = config.statistics_cache_dir / cache_key
    info_path = cache_dir / "info.json"
    statistics_path = cache_dir / "statistics.json"

    if statistics_path.exists():
        print(f"統計情報をキャッシュから読み込みました: {statistics_path}")
        statistics_dict = json.loads(statistics_path.read_text())
        return DataStatistics.from_dict(statistics_dict)

    print(f"統計情報を計算しています... (データ数: {len(datas)})")
    statistics = _calc_statistics(datas, workers=workers)

    cache_dir.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(json.dumps(statistics.to_dict(), ensure_ascii=False))
    info_path.write_text(json.dumps(info, ensure_ascii=False))

    return statistics
