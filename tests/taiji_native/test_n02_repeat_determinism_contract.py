"""`emit_taiji_n02_repeat_determinism.py` 的合同测：同材料重复出版的是"确定性"，不是地板。

四条都能为假（2026-10-18 那对逐位相同的读数就是这台仪器要判的东西）：

* 相同支——两趟 pass_id／时刻／墙钟都不同、但权重摘要与读数逐位相同 ⇒ `chain_is_deterministic…=True`
  且 `floor_informative_for_j_n02_1=False`，极差 0 **不许**被下游当噪声地板引用；
* 不同支——只要增益列动一格，极差就是真数、`floor_informative…=True`（这台仪器不是只认零的门牌）；
* 一次跑的两份拷贝（pass_id 相同）必须 REFUSE，否则"重复"是假的；
* 历史极差只从件里现算，件数不足两档时出版 `None`，不许拿单档编一个跨度出来。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "emit_taiji_n02_repeat_determinism.py"


def _module():
    spec = importlib.util.spec_from_file_location("n02_repeat_determinism_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(gain: float, *, digest: str, pass_id: str, started: str, seconds: float) -> dict:
    return {
        "stage": "done",
        "errors": [],
        "seconds": seconds,
        "started_utc": started,
        "pass_id": pass_id,
        "code_identity": {"scripts/training/run_taiji_n2b_powerup.py": "a" * 64},
        "material_fingerprint": {"source_count": 200, "source_sha256": "c" * 64},
        "window_bytes": 64,
        "night_k": 8,
        "night_pool_size": 200,
        "mother_digest": "m" + digest,
        "treated_digest": digest,
        "candidate_digest_after_load": digest,
        "material_treated_window": {"quality_mean": gain},
        "material_control_window": {"quality_mean": 0.0},
        "material_treated_whole": {"quality_mean": gain},
        "material_control_whole": {"quality_mean": 0.0},
        "surprise_modulation": {"quality_mean": gain},
        "weight_changed_by_powerup": True,
        "j_persistence": True,
        "restore_matches_mother": True,
        "rollback_disk_matches_mother": True,
        "g_n2_5_mother_roundtrip_identical": True,
        "j_n2b_prime_delta_quality": gain,
        "j_n2b_prime_delta_accuracy": gain,
        "whole_text_delta_quality": gain,
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def _pair(tmp_path, run_a: dict, run_b: dict) -> tuple[Path, Path]:
    return _write(tmp_path / "a.json", run_a), _write(tmp_path / "b.json", run_b)


def _identical_pair(tmp_path) -> tuple[Path, Path]:
    run_a = _run(0.895, digest="w", pass_id="p1", started="2026-10-18T02:36:52Z", seconds=1586.9)
    run_b = copy.deepcopy(run_a)
    run_b["pass_id"] = "p2"
    run_b["started_utc"] = "2026-10-18T03:04:13Z"
    run_b["seconds"] = 1601.8
    return _pair(tmp_path, run_a, run_b)


def test_an_identical_repeat_publishes_determinism_and_withdraws_the_floor(tmp_path) -> None:
    module = _module()
    a, b = _identical_pair(tmp_path)
    hist_a = _write(
        tmp_path / "h1.json", _run(-1.1, digest="x", pass_id="q1", started="s1", seconds=1.0)
    )
    hist_b = _write(
        tmp_path / "h2.json", _run(2.4, digest="y", pass_id="q2", started="s2", seconds=2.0)
    )
    out = tmp_path / "repeat.json"
    rc = module.main(
        [
            "--run-a",
            str(a),
            "--run-b",
            str(b),
            "--historical",
            str(hist_a),
            "--historical",
            str(hist_b),
            "--out-report",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["chain_is_deterministic_on_this_material"] is True
    assert payload["same_material_repeat_floor_delta_quality_abs"] == 0.0
    assert payload["floor_informative_for_j_n02_1"] is False
    assert payload["degenerate_reason"]
    assert payload["historical_reference"]["delta_quality_spread"] == 3.5
    assert payload["historical_reference"]["admissible_as_j_n02_1_floor"] is False


def test_a_moving_gain_column_publishes_a_real_spread_and_an_informative_floor(tmp_path) -> None:
    module = _module()
    a, b = _identical_pair(tmp_path)
    run_b = json.loads(b.read_text(encoding="utf-8"))
    run_b["j_n2b_prime_delta_quality"] = 0.395
    run_b["treated_digest"] = "other"
    run_b["material_treated_window"] = {"quality_mean": 0.395}
    run_b["surprise_modulation"] = {"quality_mean": 0.395}
    _write(b, run_b)
    out = tmp_path / "repeat.json"
    rc = module.main(["--run-a", str(a), "--run-b", str(b), "--out-report", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["same_material_repeat_floor_delta_quality_abs"] == 0.5
    assert payload["chain_is_deterministic_on_this_material"] is False
    assert payload["floor_informative_for_j_n02_1"] is True


def test_two_copies_of_one_run_are_refused(tmp_path) -> None:
    """pass_id／时刻／墙钟三列里任何一列相同 ⇒ 那不是两次独立执行，件不许出版。"""
    module = _module()
    a, b = _identical_pair(tmp_path)
    run_b = json.loads(b.read_text(encoding="utf-8"))
    run_b["pass_id"] = json.loads(a.read_text(encoding="utf-8"))["pass_id"]
    _write(b, run_b)
    out = tmp_path / "repeat.json"
    assert module.main(["--run-a", str(a), "--run-b", str(b), "--out-report", str(out)]) == 2
    assert out.exists() is False


def test_a_different_material_pile_is_refused(tmp_path) -> None:
    module = _module()
    a, b = _identical_pair(tmp_path)
    run_b = json.loads(b.read_text(encoding="utf-8"))
    run_b["material_fingerprint"] = {"source_count": 200, "source_sha256": "f" * 64}
    _write(b, run_b)
    assert (
        module.main(
            ["--run-a", str(a), "--run-b", str(b), "--out-report", str(tmp_path / "r.json")]
        )
        == 2
    )


def test_a_single_historical_report_does_not_invent_a_spread(tmp_path) -> None:
    module = _module()
    a, b = _identical_pair(tmp_path)
    hist = _write(
        tmp_path / "h1.json", _run(-1.1, digest="x", pass_id="q1", started="s1", seconds=1.0)
    )
    out = tmp_path / "repeat.json"
    assert (
        module.main(
            [
                "--run-a",
                str(a),
                "--run-b",
                str(b),
                "--historical",
                str(hist),
                "--out-report",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["historical_reference"]["delta_quality_spread"] is None


def test_a_published_repeat_artifact_is_not_clobbered(tmp_path) -> None:
    module = _module()
    a, b = _identical_pair(tmp_path)
    out = tmp_path / "repeat.json"
    assert module.main(["--run-a", str(a), "--run-b", str(b), "--out-report", str(out)]) == 0
    first = out.read_bytes()
    assert module.main(["--run-a", str(a), "--run-b", str(b), "--out-report", str(out)]) == 2
    assert out.read_bytes() == first
