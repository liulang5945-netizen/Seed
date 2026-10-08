"""DEBT-G63 的契约测：收尾行的窗口除零护栏——**零窗口必须出版 None，不许印 0.0**。

起因是实测：甲臂 `output/n3a_armA/progress_exit.json` 里 `online_accuracy=0.0`、
`mean_surprise=0.0`，而同件 `holdout_surprise=2.321688330698063` 是真数 ⇒
"这一行没有窗口样本"被 `x/max(1,0)` 抹成了"精度为零"这种看起来像能力的读数。

四支都走：①纯函数的两面（零窗口⇒None／有窗口⇒真数）；②不变式（收尾行的
`online_accuracy is None` 与 `window_ticks == 0` 同真同假）；③正常档（窗口非零）
仍出版数值且键集不缩；④helper 在模块级、缺参必须 TypeError。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.training import train_seed_corpus as trainer  # noqa: E402


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    docs = [
        "问：甲是什么？答：甲是一个符号串，用来喂这台仪器的最小语料。",
        "问：乙呢？答：乙是另一个符号串，长度与甲接近但不完全相同。",
        "问：丙呢？答：丙还是符号串，第三篇，保证窗口切分时不只剩一篇。",
    ]
    path.write_text(
        "".join(json.dumps({"text": doc}) + "\n" for doc in docs),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _run(tmp_path: Path, **over: Any) -> Path:
    from seed import SeedConfig

    progress = tmp_path / "progress.jsonl"
    trainer.run_training(
        corpus_paths=[_corpus(tmp_path)],
        config=SeedConfig(),
        epochs=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=progress,
        checkpoint_every=1_000_000,
        max_symbols=120,
        readout="predictive",
        **over,
    )
    return progress


def _exit_record(progress: Path) -> dict[str, Any]:
    exit_path = trainer.exit_record_path(progress)
    return json.loads(exit_path.read_text(encoding="utf-8"))


def test_window_readouts_pure_function_both_directions() -> None:
    assert trainer._window_readouts(0, 0.0, 0) == (None, None)
    assert trainer._window_readouts(0, 0.0, -5) == (None, None)
    accuracy, surprise = trainer._window_readouts(30, 90.0, 100)
    assert accuracy == 0.3
    assert surprise == 0.9
    #: 关键的反面：旧护栏 `x/max(1,0)` 会把零窗口算成数值 0.0，这里必须不是浮点。
    assert not isinstance(trainer._window_readouts(0, 0.0, 0)[0], float)


def test_exit_row_invariant_none_iff_zero_window(tmp_path: Path) -> None:
    progress = _run(tmp_path, progress_every=10)
    record = _exit_record(progress)
    assert record["window_ticks"] == 0, record["window_ticks"]
    assert record["online_accuracy"] is None
    assert record["mean_surprise"] is None
    #: 同件的 holdout 两列仍是真数——None 只属于窗口量，不许把整行变成空。
    assert isinstance(record["holdout_surprise"], float)
    assert isinstance(record["ticks_at_exit"], int)
    assert record["exit_reason"] == "max_symbols_reached"


def test_nonzero_window_still_publishes_numbers_and_keys_are_untouched(tmp_path: Path) -> None:
    progress = _run(tmp_path, progress_every=1_000_000)
    record = _exit_record(progress)
    assert record["window_ticks"] > 0
    assert isinstance(record["online_accuracy"], float)
    assert isinstance(record["mean_surprise"], float)
    #: 键集只许多不许少（周期性行的形状由 G14 那条守卫钉，这里复核收尾行）。
    required = {
        "epoch",
        "ticks",
        "window_ticks",
        "online_accuracy",
        "mean_surprise",
        "holdout_surprise",
        "holdout_surprise_v2",
        "elapsed_seconds",
        "exit_reason",
        "reached_budget",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(record))
    assert not missing, missing


def test_helper_is_module_level_and_typed() -> None:
    callable_obj = getattr(trainer, "_window_readouts", None)
    assert callable(callable_obj)
    with pytest.raises(TypeError):
        callable_obj()  # type: ignore[call-arg]
