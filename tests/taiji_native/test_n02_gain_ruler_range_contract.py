"""C4 增益面尺范围仪的合同测：判级词必须封闭、四条守卫必须能为假、真件必须自己走过去。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/verify_taiji_n02_gain_ruler_range.py"


def load_module():
    spec = importlib.util.spec_from_file_location("n02_gain_ruler_range", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rr = load_module()


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def powerup(**overrides) -> dict:
    base = {
        "stage": "done",
        "fail_closed_pass": True,
        "g_n2_5_mother_roundtrip_identical": True,
        "by_source": {"constraints": 0, "interactions": 200, "workbench_capabilities": 0},
        "j_n2b_prime_delta_quality": 2.5,
        "j_n2b_prime_delta_accuracy": 0.0,
        "material_treated_whole": {"items": 8.0},
        "material_control_whole": {"items": 8.0},
        "mother_digest": "a" * 64,
        "mother_tick": 273,
    }
    base.update(overrides)
    return base


def guard(**overrides) -> dict:
    base = {"guard_pass": True, "digest_before": "b" * 64, "digest_after": "b" * 64}
    base.update(overrides)
    return base


def test_channel_count_selects_a_predeclared_verdict() -> None:
    two = rr.measure(powerup(by_source={"a": 5, "b": 7, "c": 0}), guard())
    assert two[rr.RANGE_KEY] == 2 and two["range_verdict"] == "two_channels_live"
    one = rr.measure(powerup(), guard())
    assert one[rr.RANGE_KEY] == 1 and one["range_verdict"] == "single_channel_only"
    none = rr.measure(powerup(by_source={"a": 0, "b": 0}), guard())
    assert none[rr.RANGE_KEY] == 0 and none["range_verdict"] == "no_channel_live"
    #: 判级词是封闭表：三条全活时表里没有词，必须响亮拒绝，而不是给我现造一个"三通道"。
    with pytest.raises(rr.Refuse):
        rr.measure(powerup(by_source={"a": 1, "b": 2, "c": 3}), guard())


def test_control_arm_must_actually_be_a_control() -> None:
    #: 对照臂改了权重就不是对照——这一支为假时整件读数失去意义，所以守卫必须先拦。
    with pytest.raises(rr.Refuse, match="control"):
        rr.measure(powerup(), guard(digest_after="c" * 64))
    with pytest.raises(rr.Refuse, match="guard"):
        rr.measure(powerup(), guard(guard_pass=False))


def test_denominators_must_be_the_same_set() -> None:
    with pytest.raises(rr.Refuse, match="denominator"):
        rr.measure(powerup(material_control_whole={"items": 6.0}), guard())


def test_missing_admission_fields_refuse_rather_than_guess() -> None:
    for payload, keyword in (
        ({"stage": "budget-stop"}, "stage"),
        ({"fail_closed_pass": False}, "fail_closed"),
        ({"g_n2_5_mother_roundtrip_identical": False}, "roundtrip"),
        ({"by_source": {}}, "by_source"),
        ({"j_n2b_prime_delta_quality": "2.5"}, "delta_quality"),
    ):
        candidate = powerup(**payload)
        if "j_n2b_prime_delta_quality" in payload:
            candidate["j_n2b_prime_delta_quality"] = "2.5"
        with pytest.raises(rr.Refuse, match=keyword):
            rr.measure(candidate, guard())


def test_publication_carries_the_zero_work_disclosure() -> None:
    report = rr.measure(powerup(), guard())
    assert (
        report["models_loaded"] == 0
        and report["model_queries"] == 0
        and report["parameter_updates"] == 0
    )
    assert report["writes_any_face"] is False and report["is_capability_claim"] is False
    assert report["historical_delta_transferable"] is False
    assert report["format"] == "taiji-n02-gain-ruler-range-v1"


def test_the_write_target_is_one_shot(tmp_path: Path) -> None:
    up = write(tmp_path / "up.json", powerup())
    gd = write(tmp_path / "guard.json", guard())
    out = tmp_path / "range.json"
    argv = ["--powerup-report", str(up), "--guard-report", str(gd), "--out-report", str(out)]
    assert rr.main(argv) == 0
    before = out.read_bytes()
    assert rr.main(argv) == 2, "second write must refuse"
    assert out.read_bytes() == before, "refusal must not touch the published report"


def test_the_sealed_pair_reads_a_single_live_channel(tmp_path: Path) -> None:
    #: 新算法第一次执行必须打在真件上：今天这枚面的分辨率是**一条来源通道活着**，
    #: 双臂分母各 8 项，accuracy 那列的 Δ 恰为 0.0（原样出版，不相减、不重算）。
    out = tmp_path / "real.json"
    assert rr.main(["--out-report", str(out)]) == 0
    entry = json.loads(out.read_text(encoding="utf-8"))
    assert entry[rr.RANGE_KEY] == 1 and entry["range_verdict"] == "single_channel_only"
    assert entry["channels"] == {
        "constraints": 0,
        "interactions": 200,
        "workbench_capabilities": 0,
    }
    assert entry["deltas"]["j_n2b_prime_delta_accuracy"] == 0.0
    assert entry["deltas"]["j_n2b_prime_delta_quality"] == pytest.approx(2.4829673663245373)
    assert entry["denominator_items_per_arm"] == {
        "material_treated_whole": 8.0,
        "material_control_whole": 8.0,
    }
    sealed = json.loads(rr.POWERUP.read_text(encoding="utf-8"))
    assert entry["sealed_mother"] == {
        "digest": sealed["mother_digest"],
        "tick": sealed["mother_tick"],
    }
