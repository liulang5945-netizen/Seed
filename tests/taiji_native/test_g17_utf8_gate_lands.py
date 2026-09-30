"""DEBT-G17 落地守卫：`well_formed` 的"UTF-8 合法"检查必须**真能拒**。

owner 裁定（2026-09-30，PLAN-A-30 §7b-4）：G17 检查落地＋L1 参考值按上限 12/15 重录。
旧实现 `text.encode("utf-8").decode("utf-8")` 只接 `UnicodeDecodeError`，而 `text` 是 `str`
⇒ encode 除非遇到 lone surrogate（抛的是 `UnicodeEncodeError`，旧 `except` 不接＝直接炸）
否则永不抛 ⇒ **这条检查在 str 上从未拒过任何输入**（§2as 三条探针实证，含 8 个连续 U+FFFD）。

修正后必须拒的两类输入：
* lone surrogate（`"\ud800"`）——encode 抛 `UnicodeEncodeError`；
* 含 U+FFFD 的文本——decode 替换字符＝上游已有断字（口径与评分仪器的 `utf8_decodable` 同源：
  `"\ufffd" not in text`）。
其余行为（长度/单字占比/NLL 支路）逐位不变；产品副本与仪器副本仍逐位同判（漂移守卫沿用）。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import diag_taiji_r2_surface_decode as instrument  # noqa: E402

from seed import surface_gate  # noqa: E402


def _model() -> tuple:
    return surface_gate.load_surface_ngram(
        PROJECT_ROOT / "checkpoints" / "seed_surface_ngram.lzma"
    )


def test_lone_surrogate_is_rejected_not_raised() -> None:
    assert surface_gate.well_formed("\ud800这一句足够长", _model()) is False
    assert instrument.well_formed("\ud800这一句足够长", _model()) is False


def test_u_fffd_is_rejected() -> None:
    assert surface_gate.well_formed("这一句里有\ufffd替换字符", _model()) is False
    assert instrument.well_formed("这一句里有\ufffd替换字符", _model()) is False


def test_clean_text_verdicts_unchanged() -> None:
    #: 修正只加严字节合法一道；干净文本的判定路径与修正前一致（真实随包 n 元工件）。
    assert surface_gate.well_formed("这一句是干净中文，长度也够", _model()) is True
    assert surface_gate.well_formed("ab", _model()) is False  # 太短
    assert surface_gate.well_formed("君" * 60, _model()) is False  # 单字占比超限


def test_product_and_instrument_copies_still_agree_bitwise() -> None:
    pool = [
        "这一句是干净中文，长度也够",
        "词类一文词类一词文本",
        "\ud800孤代理",
        "带\ufffd的答复",
        "君" * 30,
        "",
        "ab",
    ]
    for text in pool:
        assert surface_gate.well_formed(text, _model()) == instrument.well_formed(text, _model()), (
            text
        )
