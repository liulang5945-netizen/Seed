"""PLAN-N5-05 G-N5e-4 的落地面：压强面**收尾行**必须自述"位置输入有没有被走到"。

甲把位置列接进发育 F1 通路之后，`J-N5e-4` 的续训双臂要能**从面件里**读到这件事发生没有，
而不是我事后去抄 `Seed` 的计数器——本仓的纪律是在场性按现读、不许由命令行反推
（同族先例：`mode_readings` 与 `bridge_gate_actual`，PLAN-N3-04 §3；㊵-618 那族"计数器没回来
之前不许写喂过/没喂过"）。

三支各自都能为假：

* 位置关闭＋发育装配 ⇒ `column_present=false` 且两枚计数为 0——"根本没有列"与"有列但没喂"
  必须是两种分得开的读数，否则这条自述就是恒真；
* 位置开启＋发育装配 ⇒ `column_present=true`、`probability_steps > 0 ∧ learn_steps > 0`；
* 收尾行**缺这枚键** ⇒ 当场响亮失败（取数函数拒绝，不折叠成"零"）。

夹具全部落在 pytest 的 `tmp_path`，不碰 `reports/` 与 `output/`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.training import train_seed_corpus as trainer  # noqa: E402
from seed.config import SeedConfig  # noqa: E402

#: 收尾行里那枚新披露键（缺它就等于"没测"，不许读成"没走到"）。
TAIL_KEY = "position_input"

DOCS = ("问：甲是什么？答：甲是一个符号串。", "问：乙呢？答：乙是另一个符号串。")


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps({"text": doc}, ensure_ascii=False) + "\n" for doc in DOCS),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _tail(tmp_path: Path, *, position: bool) -> dict[str, Any]:
    pressure = tmp_path / "pressure.jsonl"
    config = SeedConfig()
    if position:
        config = trainer.apply_experiment_flags(config, readout_position=True)
    trainer.run_training(
        corpus_paths=[_corpus(tmp_path)],
        config=config,
        epochs=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=tmp_path / "progress.jsonl",
        checkpoint_every=1_000_000,
        progress_every=1_000_000,
        max_symbols=120,
        readout="predictive",
        pressure_record=pressure,
        developmental_fast_slow=True,
    )
    for raw in pressure.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        if record.get("kind") == "tail":
            return record
    raise AssertionError("面里没有 tail 行")


def _read_position_disclosure(tail: dict[str, Any]) -> dict[str, int | bool]:
    """取数函数**自己**拒绝缺键：把"没测"折叠成 0 是本仓付过很多次学费的那类假读数。"""

    if TAIL_KEY not in tail:
        raise AssertionError(f"收尾行缺 {TAIL_KEY} ⇒ 位置输入是否被走到没测，不判")
    report = dict(tail[TAIL_KEY])
    assert set(report) == {"column_present", "probability_steps", "learn_steps"}, report
    return report


def test_position_off_arm_publishes_an_absent_column_not_a_fake_zero(tmp_path: Path) -> None:
    report = _read_position_disclosure(_tail(tmp_path, position=False))
    assert report == {"column_present": False, "probability_steps": 0, "learn_steps": 0}, report


def test_position_on_arm_publishes_both_walked_counters(tmp_path: Path) -> None:
    tail = _tail(tmp_path, position=True)
    #: 装配自述必须同时在场：这条读数的意义是"发育通路开着且位置列被走到"，两半缺一就不成立。
    assert tail["learning_mode_at_close"] == "fast_slow", tail
    report = _read_position_disclosure(tail)
    assert report["column_present"] is True, report
    assert int(report["probability_steps"]) > 0, report
    assert int(report["learn_steps"]) > 0, report


def test_a_tail_without_the_key_is_refused_not_read_as_zero() -> None:
    #: 反面对照：把键拿掉 ⇒ 取数函数必须拒绝，而不是给出"两枚计数都是 0"那种看着像结论的读数。
    tail = {"kind": "tail", "learning_mode_at_close": "fast_slow"}
    try:
        _read_position_disclosure(tail)
    except AssertionError as error:
        assert "没测" in str(error), str(error)
    else:
        raise AssertionError("缺披露键却没被拒绝 ⇒ 这条自述能为假的前提没了")
