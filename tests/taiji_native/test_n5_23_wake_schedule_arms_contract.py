"""PLAN-N5-12 批次 B 的契约册：双臂取数件的两条出口都要能被打到。

要紧的三条：
* 判读器的**两支都要走**（只验 fail-closed 等于没验）——这里既测 `arms_separable`，
  也测退化流下的 `ruler_unusable` 与 `arm_not_actually_two`；
* 尺的可用性不许拿对照臂的翻转数当动态范围（`fixed_wake` 臂按定义恒 0）；
* 判据只按已冻那一档噪声幅度出值，逐路步尺的梯只作披露，不许事后换档凑结论。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "measure_taiji_n5_12_wake_schedule_arms.py"


def _load():
    spec = importlib.util.spec_from_file_location("n5_12_wake_arms", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verdicts_are_deterministic_for_a_given_seed() -> None:
    module = _load()
    first = module.measure(seed=7)
    second = module.measure(seed=7)
    assert first["flips"] == second["flips"]
    assert first["verdicts"] == second["verdicts"]


def test_zero_amplitude_perturbation_is_the_identity() -> None:
    module = _load()
    stream = module.clean_stream()
    assert module.perturb(stream, seed=7, amplitude=0.0) == stream


def test_frozen_amplitude_equals_the_streams_own_max() -> None:
    module = _load()
    stream = module.clean_stream()
    assert module.frozen_amplitude(stream) == max(v for row in stream for v in row)
    assert module.noise_units(stream) != (module.frozen_amplitude(stream),) * len(stream)


def test_degenerate_stream_exercises_both_fail_closed_exits() -> None:
    """一条不动的流：翻转恒 0 ⇒ J-N5l-2 记 `ruler_unusable`，
    而两臂逐位相同 ⇒ J-N5l-4 必须报 `arm_not_actually_two`（这支不被可用性挡住，
    否则它永远走不到＝惰性出口）。"""

    module = _load()
    module.clean_stream = lambda: [(0.0, 0.0, 0.0, 0.0)] * module.STEPS
    report = module.measure(seed=7)
    assert report["verdicts"]["ruler_usable"] is False
    assert report["verdicts"]["j_n5l_2_noise_rewake"] == "ruler_unusable"
    assert report["verdicts"]["j_n5l_4_fixed_schedule_control"] == "arm_not_actually_two"
    assert report["flips"]["adaptive_clean"] == 0
    assert report["flips"]["fixed_wake_clean"] == 0


def test_real_stream_gives_a_usable_ruler_and_separable_arms() -> None:
    module = _load()
    report = module.measure(seed=7)
    verdicts = report["verdicts"]
    assert verdicts["ruler_usable"] is True
    assert report["flips"]["adaptive_clean"] > 0
    assert report["flips"]["fixed_wake_clean"] == 0
    assert verdicts["j_n5l_4_fixed_schedule_control"] == "arms_separable"
    assert verdicts["j_n5l_2_noise_rewake"].startswith(
        ("noise_does_not_re_wake", "noise_changes_scheduling:")
    )


def test_the_ladder_is_disclosure_not_the_verdict() -> None:
    module = _load()
    report = module.measure(seed=7)
    ladder = report["supplementary_noise_ladder"]
    assert ladder["used_for_verdict"] is False
    assert ladder["rows"]
    assert report["fixture"]["frozen_amplitude_from_stream_max"] == 1.0
    for row in ladder["rows"]:
        assert row["amplitude"] <= report["fixture"]["frozen_amplitude_from_stream_max"]


def test_short_fixture_is_refused_loudly() -> None:
    module = _load()
    module.STEPS = 1
    with pytest.raises(ValueError):
        module.measure(seed=7)


def test_existing_target_is_refused_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    module = _load()
    target = tmp_path / "arms.json"
    target.write_text('{"sentinel": 1}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog", "--out-report", str(target), "--seed", "7"])
    assert module.main() == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"sentinel": 1}
    monkeypatch.setattr(sys, "argv", ["prog", "--out-report", str(target), "--overwrite"])
    assert module.main() == 0
    assert json.loads(target.read_text(encoding="utf-8"))["format"] == module.REPORT_FORMAT
