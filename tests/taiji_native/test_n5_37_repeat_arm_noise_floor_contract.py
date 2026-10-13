"""`audit_taiji_n5_23_repeat_arm_noise_floor.py` 的契约册（J-N5v-2／J-N5v-3 的机检形）。

判据函数必须在合成阈值表上**两支都跑过**：地板离开与不离开各一次、地板恰为零一次、
指标缺一枚一次、配置不齐一次、没跑到预算一次。这样「仪器正确」与「仪器跑过一次」
被分开证明——真读数还要等重复臂退出，本册只证仪器。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_n5_23_repeat_arm_noise_floor.py"


def _load():
    spec = importlib.util.spec_from_file_location("n5_23_floor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FLOOR = _load()


def _run_report(
    surprise: float,
    accuracy: float,
    *,
    consumed: int = 8_900_000,
    feed_every: int = 50,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "base_checkpoint_sha256": "sha-base",
        "symbols_budget": 8_900_000,
        "skip_symbols": 11_199_800,
        "fresh": True,
        "generated_feed": {"every_symbols": feed_every},
        "last_progress": {
            "symbols_budget": 8_900_000,
            "symbols_consumed": consumed,
            "cumulative_mean_surprise": surprise,
            "cumulative_online_accuracy": accuracy,
        },
    }


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    return path


def _args(tmp_path: Path, four: list[dict[str, Any]], out_name: str) -> list[str]:
    paths = [_write(tmp_path / f"arm{index}.json", payload) for index, payload in enumerate(four)]
    return [
        "--batchc-treated",
        str(paths[0]),
        "--batchc-control",
        str(paths[1]),
        "--repeat-treated",
        str(paths[2]),
        "--repeat-control",
        str(paths[3]),
        "--out-report",
        str(tmp_path / out_name),
    ]


def test_a_floor_the_observed_difference_leaves_is_supported(tmp_path: Path) -> None:
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255),
        _run_report(6.05, 0.261),
    ]
    rc = FLOOR.main(_args(tmp_path, four, "r.json"))
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["verdict"] == "holds"
    surprise = payload["floors"]["cumulative_mean_surprise"]
    #: 浮点末位：7.0−6.9 在二进制里不是恰 0.1，断言按**近似**取，不为了凑整数改判据。
    assert surprise["floor"] == pytest.approx(0.1, abs=1e-9)
    assert surprise["observed_difference_batchc"] == pytest.approx(1.0, abs=1e-9)
    assert surprise["verdict"] == "magnitude_supported"


def test_a_floor_the_difference_does_not_leave_is_blocked(tmp_path: Path) -> None:
    #: 这一支同时钉住「恰好等于地板」那一格：观察差 1.0、地板 1.0 ⇒ **不算离开**（严格大于）。
    #: 少了这一支，`observed > floor` 会被悄悄写成 `>=`，而两者在真读数上常常就差最后一位。
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(8.0, 0.25),
        _run_report(5.0, 0.26),
    ]
    rc = FLOOR.main(_args(tmp_path, four, "b.json"))
    payload = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert rc == 0
    surprise = payload["floors"]["cumulative_mean_surprise"]
    assert surprise["floor"] == pytest.approx(1.0, abs=1e-9)
    assert surprise["observed_difference_batchc"] == pytest.approx(1.0, abs=1e-9)
    assert surprise["verdict"] == "magnitude_blocked"
    assert payload["verdicts"]["any_metric_leaves_its_floor"] is False


def test_a_zero_floor_is_unjudged_not_small(tmp_path: Path) -> None:
    same = _run_report(7.0, 0.25)
    other = _run_report(6.0, 0.26)
    four = [same, other, json.loads(json.dumps(same)), json.loads(json.dumps(other))]
    rc = FLOOR.main(_args(tmp_path, four, "z.json"))
    payload = json.loads((tmp_path / "z.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["floors"]["cumulative_mean_surprise"]["floor"] == 0.0
    assert payload["floors"]["cumulative_mean_surprise"]["verdict"] == "no_spread_unjudged"
    assert payload["verdicts"]["any_metric_leaves_its_floor"] is False


def test_an_arm_that_stopped_short_cannot_price_the_spread(tmp_path: Path) -> None:
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255, consumed=8_000_000),
        _run_report(6.05, 0.261),
    ]
    rc = FLOOR.main(_args(tmp_path, four, "s.json"))
    payload = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["verdict"] == "rejected"
    assert payload["failing_verdicts"] == ["all_four_arms_at_budget"]


def test_a_fourth_arm_is_not_a_repeat(tmp_path: Path) -> None:
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255, feed_every=25),
        _run_report(6.05, 0.261),
    ]
    rc = FLOOR.main(_args(tmp_path, four, "c.json"))
    payload = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["failing_verdicts"] == ["same_configuration_within_role"]


def test_a_missing_metric_column_means_no_floor_not_zero(tmp_path: Path) -> None:
    broken = _run_report(7.0, 0.25)
    del broken["last_progress"]["cumulative_online_accuracy"]
    four = [broken, _run_report(6.0, 0.26), _run_report(6.9, 0.255), _run_report(6.05, 0.261)]
    rc = FLOOR.main(_args(tmp_path, four, "m.json"))
    payload = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["floors"]["cumulative_online_accuracy"]["verdict"] == "metric_absent"
    assert payload["floors"]["cumulative_online_accuracy"].get("floor") is None
    assert payload["failing_verdicts"] == ["floors_formed"]


def test_cap_dimensions_get_their_own_floor(tmp_path: Path) -> None:
    batch = {
        "dimensions": [
            {"dimension": "D", "delta": -1, "enters_verdict": True},
            {"dimension": "C", "delta": 0, "enters_verdict": True},
            {"dimension": "B", "delta": 5, "enters_verdict": False},
        ]
    }
    repeat = {
        "dimensions": [
            {"dimension": "D", "delta": 2, "enters_verdict": True},
            {"dimension": "C", "delta": 0, "enters_verdict": True},
        ]
    }
    paths = [_write(tmp_path / "capb.json", batch), _write(tmp_path / "capr.json", repeat)]
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255),
        _run_report(6.05, 0.261),
    ]
    rc = FLOOR.main(
        _args(tmp_path, four, "cap.json")
        + ["--cap-pairing-batchc", str(paths[0]), "--cap-pairing-repeat", str(paths[1])]
    )
    payload = json.loads((tmp_path / "cap.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["cap"]["dimensions"] == ["C", "D"]
    assert payload["cap"]["per_dimension"]["D"]["floor"] == 3
    assert payload["cap"]["per_dimension"]["C"]["floor"] == 0


def test_missing_input_wrong_kind_and_overwrite_guard(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.json", _run_report(7.0, 0.25))
    not_run = _write(tmp_path / "not_run.json", {"format": "something-else"})
    base = [
        "--batchc-treated",
        str(good),
        "--batchc-control",
        str(good),
        "--repeat-treated",
        str(good),
        "--repeat-control",
        str(good),
        "--out-report",
        str(tmp_path / "never.json"),
    ]
    assert FLOOR.main(base) == 0
    assert FLOOR.main(base) == 2
    missing = list(base)
    missing[1] = str(tmp_path / "gone.json")
    assert FLOOR.main(missing) == 2
    wrong_kind = list(base)
    wrong_kind[1] = str(not_run)
    assert FLOOR.main(wrong_kind) == 2
