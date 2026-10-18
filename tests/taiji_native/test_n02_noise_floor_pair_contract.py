"""`emit_taiji_n02_noise_floor_from_pair.py` 的合同测：地板只有在"两趟真是同一配置的两趟"时才出版。

三条都能为假（这是 DEBT-G168 那笔代价的形状：历史两趟没有代码身份，极差 2.5 只能当披露）：

* 成功支——两趟身份逐字相同 ⇒ 出版 `noise_floor_delta_quality_abs` 与它的分子分母；
* 拒绝支——身份不同／材料指纹不同／中阶段／权重逐位相同／已有成品件，各自必须 rc=2 且不落件；
* 反面对照——把某一列改坏就必然红（若整面测在"身份随便改"下仍绿，仪器等于没守）。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "emit_taiji_n02_noise_floor_from_pair.py"

FLOOR_KEY = "noise_floor_delta_quality_abs"


def _module():
    spec = importlib.util.spec_from_file_location("n02_noise_floor_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_run(gain: float, *, digest: str, pass_id: str) -> dict:
    return {
        "format": "taiji-n2b-powerup-v1",
        "stage": "done",
        "errors": [],
        "seconds": 1390.0,
        "code_identity": {
            "scripts/training/run_taiji_n2b_powerup.py": "a" * 64,
            "scripts/training/run_taiji_n2_powerup.py": "b" * 64,
        },
        "material_fingerprint": {
            "source_count": 200,
            "source_sha256": "c" * 64,
            "night_selected_count": 8,
            "night_selected_sha256": "d" * 64,
        },
        "window_bytes": 64,
        "night_k": 8,
        "night_pool_size": 200,
        "projection_records": 200,
        "pass_id": pass_id,
        "treated_digest": digest,
        "weight_changed_by_powerup": True,
        "j_persistence": True,
        "restore_matches_mother": True,
        "rollback_disk_matches_mother": True,
        "g_n2_5_mother_roundtrip_identical": True,
        "j_n2b_prime": "holds" if gain > 0 else "not_holds",
        "j_n2b_prime_delta_quality": gain,
        "j_n2b_prime_delta_accuracy": gain / 2.0,
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def _pair(tmp_path, run_a: dict, run_b: dict, sealed: dict) -> tuple[Path, Path, Path]:
    return (
        _write(tmp_path / "run_a.json", run_a),
        _write(tmp_path / "run_b.json", run_b),
        _write(tmp_path / "sealed.json", sealed),
    )


def test_a_valid_pair_publishes_the_floor_with_its_units(tmp_path) -> None:
    module = _module()
    run_a = _base_run(0.4, digest="w1", pass_id="p1")
    run_b = _base_run(0.1, digest="w2", pass_id="p2")
    sealed = _base_run(0.2, digest="w0", pass_id="p0")
    a, b, s = _pair(tmp_path, run_a, run_b, sealed)
    out = tmp_path / "floor.json"
    rc = module.main(
        ["--run-a", str(a), "--run-b", str(b), "--sealed", str(s), "--out-report", str(out)]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[FLOOR_KEY] == 0.30000000000000004
    assert payload["noise_floor_delta_accuracy_abs"] == 0.15000000000000002
    assert payload["pair_size"] == 2
    assert payload["shape_matches_sealed_pair"] is True
    assert payload["criteria"]["J_N02_1_floor_units"]
    assert payload["identity"]["material_fingerprint"]["source_sha256"] == "c" * 64
    assert payload["identity"]["material_fingerprint"]["night_selected_sha256"] == "d" * 64


def test_different_code_identity_refuses_and_writes_no_artifact(tmp_path) -> None:
    module = _module()
    run_a = _base_run(0.4, digest="w1", pass_id="p1")
    run_b = copy.deepcopy(run_a)
    run_b["treated_digest"] = "w2"
    run_b["pass_id"] = "p2"
    run_b["code_identity"]["scripts/training/run_taiji_n2b_powerup.py"] = "e" * 64
    a, b, s = _pair(tmp_path, run_a, run_b, _base_run(0.2, digest="w0", pass_id="p0"))
    out = tmp_path / "floor.json"
    assert (
        module.main(
            ["--run-a", str(a), "--run-b", str(b), "--sealed", str(s), "--out-report", str(out)]
        )
        == 2
    )
    assert out.exists() is False


def test_different_material_pile_refuses(tmp_path) -> None:
    module = _module()
    run_a = _base_run(0.4, digest="w1", pass_id="p1")
    run_b = copy.deepcopy(run_a)
    run_b["treated_digest"] = "w2"
    run_b["pass_id"] = "p2"
    run_b["material_fingerprint"]["source_sha256"] = "f" * 64
    a, b, s = _pair(tmp_path, run_a, run_b, _base_run(0.2, digest="w0", pass_id="p0"))
    assert (
        module.main(
            [
                "--run-a",
                str(a),
                "--run-b",
                str(b),
                "--sealed",
                str(s),
                "--out-report",
                str(tmp_path / "f.json"),
            ]
        )
        == 2
    )


def test_a_run_that_did_not_really_sleep_refuses(tmp_path) -> None:
    """`weight_changed_by_powerup` 为假的趟不参与地板：它测的是"没睡"的抖动，不是这条链的。"""
    module = _module()
    run_a = _base_run(0.4, digest="w1", pass_id="p1")
    run_a["weight_changed_by_powerup"] = False
    run_b = _base_run(0.1, digest="w2", pass_id="p2")
    a, b, s = _pair(tmp_path, run_a, run_b, _base_run(0.2, digest="w0", pass_id="p0"))
    assert (
        module.main(
            [
                "--run-a",
                str(a),
                "--run-b",
                str(b),
                "--sealed",
                str(s),
                "--out-report",
                str(tmp_path / "f.json"),
            ]
        )
        == 2
    )


def test_identical_post_sleep_weights_refuses_a_zero_floor(tmp_path) -> None:
    module = _module()
    run_a = _base_run(0.4, digest="same", pass_id="p1")
    run_b = _base_run(0.4, digest="same", pass_id="p2")
    a, b, s = _pair(tmp_path, run_a, run_b, _base_run(0.2, digest="w0", pass_id="p0"))
    out = tmp_path / "floor.json"
    assert (
        module.main(
            ["--run-a", str(a), "--run-b", str(b), "--sealed", str(s), "--out-report", str(out)]
        )
        == 2
    )
    assert out.exists() is False


def test_mid_stage_report_refuses(tmp_path) -> None:
    module = _module()
    run_a = _base_run(0.4, digest="w1", pass_id="p1")
    run_a["stage"] = "treated_read"
    run_b = _base_run(0.1, digest="w2", pass_id="p2")
    a, b, s = _pair(tmp_path, run_a, run_b, _base_run(0.2, digest="w0", pass_id="p0"))
    assert (
        module.main(
            [
                "--run-a",
                str(a),
                "--run-b",
                str(b),
                "--sealed",
                str(s),
                "--out-report",
                str(tmp_path / "f.json"),
            ]
        )
        == 2
    )


def test_a_published_floor_artifact_is_not_clobbered(tmp_path) -> None:
    module = _module()
    run_a = _base_run(0.4, digest="w1", pass_id="p1")
    run_b = _base_run(0.1, digest="w2", pass_id="p2")
    a, b, s = _pair(tmp_path, run_a, run_b, _base_run(0.2, digest="w0", pass_id="p0"))
    out = tmp_path / "floor.json"
    assert (
        module.main(
            ["--run-a", str(a), "--run-b", str(b), "--sealed", str(s), "--out-report", str(out)]
        )
        == 0
    )
    first = out.read_bytes()
    assert (
        module.main(
            ["--run-a", str(a), "--run-b", str(b), "--sealed", str(s), "--out-report", str(out)]
        )
        == 2
    )
    assert out.read_bytes() == first


def test_shape_mismatch_with_the_sealed_pair_is_published_not_silently_dropped(tmp_path) -> None:
    """剂量形状与封存那对不同 ⇒ 件照样出版，但把"不可搬阈值"这一列落成 False 供引用方判。"""
    module = _module()
    run_a = _base_run(0.4, digest="w1", pass_id="p1")
    run_b = _base_run(0.1, digest="w2", pass_id="p2")
    sealed = _base_run(0.2, digest="w0", pass_id="p0")
    sealed["night_k"] = 4
    a, b, s = _pair(tmp_path, run_a, run_b, sealed)
    out = tmp_path / "floor.json"
    assert (
        module.main(
            ["--run-a", str(a), "--run-b", str(b), "--sealed", str(s), "--out-report", str(out)]
        )
        == 0
    )
    assert json.loads(out.read_text(encoding="utf-8"))["shape_matches_sealed_pair"] is False
