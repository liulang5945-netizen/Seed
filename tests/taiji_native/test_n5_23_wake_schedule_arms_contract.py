"""PLAN-N5-12 批次 B 的契约册：双臂取数件的两条出口都要能被打到。

要紧的三条：
* 判读器的**两支都要走**（只验 fail-closed 等于没验）——这里既测 `arms_separable`，
  也测退化流下的 `ruler_unusable` 与 `arm_not_actually_two`；
* 尺的可用性不许拿对照臂的翻转数当动态范围（`fixed_wake` 臂按定义恒 0）；
* 判据只按已冻那一档噪声幅度出值，逐路步尺的梯只作披露，不许事后换档凑结论。
"""

from __future__ import annotations

import ast
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
    """退化流上两臂逐位相同＝『其实只有一臂』，且 `measure()` 必须响亮拒绝出值。

    已冻的单点式在这条流上会读成 `noise_does_not_re_wake`（扰动被每一路的取值范围夹死，
    副本＝原流）——那正是"守卫不能为假"的反面，所以 `measure()` 改成先判空再出值。
    """

    module = _load()
    flat = [(0.0, 0.0, 0.0, 0.0)] * module.STEPS
    controller = module.HomeostaticController()
    adaptive, _ = module.run_arm(controller, flat, arm="adaptive")
    fixed, _ = module.run_arm(controller, flat, arm="fixed_wake")
    assert adaptive == fixed
    assert module._count_flips(adaptive) == 0
    module.clean_stream = lambda: flat
    with pytest.raises(ValueError, match="stream_degenerate"):
        module.measure(seed=7)


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
    ladder = report["noise_scale_ladder"]
    assert "J-N5l-6" in ladder["used_for_verdict"]
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


def test_degenerate_stream_cannot_read_as_a_pass() -> None:
    """PLAN-N5-13 G-N5m-1 的形状：常值流必须判 `stream_degenerate` 并响亮失败。

    已冻的单点式在这条流上会读成 `noise_does_not_re_wake`（扰动被每一路的取值范围夹死，
    副本＝原流）——那正是"守卫不能为假"的反面，所以本件把它做成必红。
    """

    module = _load()
    module.clean_stream = lambda: [(0.5, 0.5, 0.5, 0.5)] * module.STEPS
    with pytest.raises(ValueError, match="stream_degenerate"):
        module.measure(seed=7)


def test_worst_case_rejects_the_favourable_scale() -> None:
    """G-N5m-1：有一档相等而另一档不等 ⇒ 必须判不成立，不许挑那一档。"""

    module = _load()
    rows = [{"amplitude": 0.25, "flips": 13.0}, {"amplitude": 1.0, "flips": 15.0}]
    assert module.worst_case_verdicts(13, rows) == "noise_changes_scheduling:13->15"
    all_equal = [{"amplitude": 0.25, "flips": 13.0}, {"amplitude": 1.0, "flips": 13.0}]
    assert module.worst_case_verdicts(13, all_equal) == "noise_does_not_re_wake"


def test_ladder_is_deterministic_and_includes_the_frozen_scale() -> None:
    """G-N5m-2 的可核形式：梯由流唯一决定，且一定含已冻那一档；mixed 不在里面。"""

    module = _load()
    stream = module.clean_stream()
    ladder = module.scale_ladder(stream)
    assert ladder == tuple(sorted(ladder))
    assert module.frozen_amplitude(stream) in ladder
    assert len(ladder) == len(set(ladder))
    assert all(scale > 0.0 for scale in ladder)
    assert len(ladder) == len(module.raw_diff_scales(stream) | {module.frozen_amplitude(stream)})


def test_verdict_expression_introduces_no_new_numeric_threshold() -> None:
    """G-N5m-3：**比较式**的右值只许是 0 与 1（夹具参数住在 `clean_stream`，不住在判定里）。

    取法收窄到 `ast.Compare` 的比较对象：第一版扫全部 `Constant` 会把 `row[0]`／`row[3]` 这种
    **下标字面量**也算成阈值（`perturb` 当场被误报两枚），那是"守卫过宽会误伤"的又一形。
    """

    module = _load()
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for name in ("worst_case_verdicts", "raw_diff_scales", "scale_ladder"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            for comparator in node.comparators:
                if (
                    isinstance(comparator, ast.Constant)
                    and isinstance(comparator.value, (int, float))
                    and comparator.value not in (0, 1, 0.0)
                ):
                    offenders.append(f"{name}:{comparator.value!r}")
    assert offenders == [], offenders
