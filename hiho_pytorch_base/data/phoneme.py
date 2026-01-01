"""音素データ処理モジュール"""

from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Self

import numpy


class BasePhoneme:
    """基底音素クラス"""

    phoneme_list: tuple[str, ...]
    num_phoneme: int
    space_phoneme: str

    def __init__(
        self,
        phoneme: str,
        start: float,
        end: float,
    ):
        self.phoneme = phoneme
        self.start = start
        self.end = end

    def __repr__(self):  # noqa: D105
        return f"Phoneme(phoneme='{self.phoneme}', start={self.start}, end={self.end})"

    def __eq__(self, o: object):  # noqa: D105
        return isinstance(o, BasePhoneme) and (
            self.phoneme == o.phoneme and self.start == o.start and self.end == o.end
        )

    def verify(self):
        """音素データの検証を行う"""
        assert self.start < self.end, f"{self} start must be less than end"
        assert self.phoneme in self.phoneme_list, f"{self} is not defined."

    @property
    def phoneme_id(self):
        """音素IDを取得する"""
        return self.phoneme_list.index(self.phoneme)

    @property
    def duration(self):
        """音素の継続時間を取得する"""
        return self.end - self.start

    @property
    def onehot(self):
        """ohehotベクトルを取得する"""
        array = numpy.zeros(self.num_phoneme, dtype=bool)
        array[self.phoneme_id] = True
        return array

    @classmethod
    def parse(cls, s: str):
        """文字列から音素データを解析する"""
        words = s.split()
        return cls(
            start=float(words[0]),
            end=float(words[1]),
            phoneme=words[2],
        )

    @classmethod
    @abstractmethod
    def convert(cls, phonemes: list[Self]) -> list[Self]:
        """音素リストを変換する"""
        pass

    @classmethod
    def verify_list(cls: type[Self], phonemes: list[Self]):
        """音素リストの検証する"""
        assert phonemes[0].start == 0, f"{phonemes[0]} start must be 0."
        for phoneme in phonemes:
            phoneme.verify()
        for pre, post in zip(phonemes[:-1], phonemes[1:], strict=False):
            assert pre.end == post.start, f"{pre} and {post} must be continuous."

    @classmethod
    def loads_julius_list(cls, text: str, verify=True):
        """テキストからJulius形式の音素リストを読み込む"""
        phonemes = [cls.parse(s) for s in text.split("\n") if len(s) > 0]
        phonemes = cls.convert(phonemes)

        if verify:
            try:
                cls.verify_list(phonemes)
            except Exception:
                raise
        return phonemes

    @classmethod
    def load_julius_list(cls, path: PathLike, verify=True):
        """ファイルからJulius形式の音素リストを読み込む"""
        try:
            phonemes = cls.loads_julius_list(Path(path).read_text(), verify=verify)
        except Exception:
            print(f"{path} is not valid.")
            raise
        return phonemes

    @classmethod
    def save_julius_list(cls, phonemes: list[Self], path: PathLike, verify=True):
        """Julius形式の音素リストをファイルに保存する"""
        if verify:
            try:
                cls.verify_list(phonemes)
            except Exception:
                print(f"{path} is not valid.")
                raise

        text = "\n".join([f"{p.start:.4f}\t{p.end:.4f}\t{p.phoneme}" for p in phonemes])
        Path(path).write_text(text)


class ArpaPhoneme(BasePhoneme):
    """ARPABET音素クラス"""

    vowel_phonemes = (
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    )

    consonant_phonemes = (
        "B",
        "CH",
        "D",
        "DH",
        "F",
        "G",
        "HH",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    )

    phoneme_list = ("pau",) + vowel_phonemes + consonant_phonemes + ("spn",)

    num_phoneme = len(phoneme_list)
    space_phoneme = "pau"

    def __init__(
        self,
        phoneme: str,
        start: float,
        end: float,
        stress: int | None = None,
    ):
        super().__init__(phoneme, start, end)
        self.stress = stress

    @classmethod
    def is_vowel(cls, phoneme_name: str) -> bool:
        """音素が母音かどうかを判定する"""
        return phoneme_name in cls.vowel_phonemes

    @classmethod
    def convert(cls, phonemes: list[Self]) -> list[Self]:  # type: ignore
        """音素リストを正規化し、ストレス情報を分離・設定する"""
        for phoneme in phonemes:
            # ストレス情報の分離と設定
            original_phoneme = phoneme.phoneme

            # 特殊音素の処理（pau, spnはストレス情報なし）
            if original_phoneme in ["pau", "spn"]:
                phoneme.stress = None
            elif original_phoneme.endswith(("0", "1", "2")):
                # ストレス情報付き音素の処理
                phoneme.phoneme = original_phoneme[:-1]
                phoneme.stress = int(original_phoneme[-1])
            else:
                # その他（ストレス情報なし）
                phoneme.stress = None

            # 従来の音素名正規化処理
            if "sil" in phoneme.phoneme:
                phoneme.phoneme = cls.space_phoneme
            elif phoneme.phoneme == "(.)":
                phoneme.phoneme = "pau"

        return phonemes
