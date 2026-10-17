"""PLAN-N-02 判读器的合同测：四条判据与四条准入守卫都必须**能为假**，且判级只从本函数出。

零模型加载、零写入面：夹具全落 `tmp_path`。另有一支跑在**真实封存件**上，用来出版"C4 今天
到底缺哪一列"——那条结论必须由仪器说出，不由散文说出（DEBT-G153 的形状）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import adjudicate_taiji_n02_c4_net_gain as c4  # noqa: E402


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def face(tmp_path: Path, name: str, extra: dict | None = None) -> Path:
    return write(tmp_path / name, {"identity": "same-face", **(extra or {})})


def measured_separation(overlap: int = 0) -> dict:
    """封存件里这条守卫的真实词汇：状态 `measured`，命中数在 `result.windows_found_in_corpus`。"""
    return {
        "status": "measured",
        "manifest_shape": "dimensions_items",
        "retention_windows": 116,
        "skipped_short_rows": 35,
        "result": {
            "probe_windows": 116,
            "windows_found_in_corpus": overlap,
            "hit_rate": overlap / 116,
            "window_bytes": 24,
        },
    }


def make_pair(
    tmp_path: Path,
    per_column: dict,
    guards: dict | None = None,
    capacity: int = 7,
) -> Path:
    faces = {
        "cap0_before": face(tmp_path, "cap0_before.json"),
        "cap0_after": face(tmp_path, "cap0_after.json"),
        "replay_before": face(tmp_path, "replay_before.json"),
        "replay_after": face(tmp_path, "replay_after.json"),
    }
    pair = {
        "format": c4.RETENTION_FORMAT,
        "prereg": "PLAN-N2-04 §2/§3/§4",
        "per_column": per_column,
        "criteria": {"columns_compared": capacity},
        "guards": (
            guards
            if guards is not None
            else {
                "G_N2c_3_cap0_same_source": {"status": "ok", "conflicts": []},
                "G_N2c_3_replay_same_source": {"status": "ok", "conflicts": []},
                "G_N2c_4_disjointness": measured_separation(),
            }
        ),
    }
    for key, path in faces.items():
        pair[key + "_path"] = str(path)
        pair[key + "_sha256"] = c4.sha(path)
    return write(tmp_path / "pair.json", pair)


def columns(**deltas: float) -> dict:
    return {
        name: {"before": 5, "after": 5 + delta, "delta": delta, "dropped": delta < 0}
        for name, delta in deltas.items()
    }


def make_gain(tmp_path: Path, delta_quality: float, delta_accuracy: float = 0.0) -> Path:
    return write(
        tmp_path / "gain.json",
        {
            c4.GAIN_QUALITY_KEY: delta_quality,
            c4.GAIN_ACCURACY_KEY: delta_accuracy,
        },
    )


def make_floor(tmp_path: Path, value: float) -> Path:
    return write(tmp_path / "floor.json", {c4.FLOOR_QUALITY_KEY: value})


def make_range_face(tmp_path: Path, usable: int = 7) -> Path:
    return write(tmp_path / "rangeface.json", {"usable_endpoints": usable})


def test_retention_criterion_has_both_branches(tmp_path: Path) -> None:
    holds = make_pair(tmp_path, columns(a=0, b=1, c=0))
    assert c4.retention_judgement(c4.load_pair(holds))["verdict"] == "retention_holds"
    slips = make_pair(tmp_path, columns(a=0, b=-1, c=2), capacity=3)
    block = c4.retention_judgement(c4.load_pair(slips))
    assert block["verdict"] == "cost_persists"
    assert block["dropped_columns"] == ["b"], block


def test_gain_uses_the_floor_and_the_boundary_is_not_resolved(tmp_path: Path) -> None:
    pair = make_pair(tmp_path, columns(a=0, b=0))
    ruler = {"state": "present", "checked_face": "x", "key": c4.RANGE_KEY, "value": 7}

    no_floor = c4.adjudicate(
        c4.load_pair(pair),
        c4.load_gain(make_gain(tmp_path, 3.0)),
        None,
        ruler,
    )
    assert no_floor["status"] == "not_judged", no_floor["status"]

    below = c4.adjudicate(
        c4.load_pair(pair),
        c4.load_gain(make_gain(tmp_path, 1.0)),
        c4.floor_quality(make_floor(tmp_path, 2.0)),
        ruler,
    )
    assert below["status"] == "not_resolved", below["status"]

    exactly = c4.adjudicate(
        c4.load_pair(pair),
        c4.load_gain(make_gain(tmp_path, 2.0)),
        c4.floor_quality(make_floor(tmp_path, 2.0)),
        ruler,
    )
    assert exactly["status"] == "not_resolved", "恰好等于地板那一支不算越线"
    assert exactly["criteria"]["J_N02_1_gain"]["noise_floor"] == 2.0

    over = c4.adjudicate(
        c4.load_pair(pair),
        c4.load_gain(make_gain(tmp_path, 2.001)),
        c4.floor_quality(make_floor(tmp_path, 2.0)),
        ruler,
    )
    assert over["status"] == "net_gain_candidate", over["status"]


def test_the_ruler_precondition_blocks_the_conjunction(tmp_path: Path) -> None:
    """P-尺：增益面没有动态范围列时，四条全绿也只能停在 range_unverified。"""
    pair = make_pair(tmp_path, columns(a=0, b=0))
    verdict = c4.adjudicate(
        c4.load_pair(pair),
        c4.load_gain(make_gain(tmp_path, 9.0)),
        c4.floor_quality(make_floor(tmp_path, 1.0)),
        c4.range_column(None),
    )
    assert verdict["status"] == "range_unverified", verdict
    assert verdict["ruler_precondition"]["key"] == c4.RANGE_KEY


def test_separation_and_comparability_each_can_break_the_conjunction(tmp_path: Path) -> None:
    ruler = {"state": "present", "checked_face": "x", "key": c4.RANGE_KEY, "value": 7}
    args = (make_gain(tmp_path, 9.0), make_floor(tmp_path, 1.0))

    unverified = make_pair(
        tmp_path,
        columns(a=0, b=0),
        guards={
            "G_N2c_3_cap0_same_source": {"status": "ok", "conflicts": []},
            "G_N2c_3_replay_same_source": {"status": "ok", "conflicts": []},
            "G_N2c_4_disjointness": {"status": "unverified", "reason": "digest unreadable"},
        },
    )
    verdict = c4.adjudicate(
        c4.load_pair(unverified), c4.load_gain(args[0]), c4.floor_quality(args[1]), ruler
    )
    assert verdict["status"] == "separation_unverified", verdict

    conflict = make_pair(
        tmp_path,
        columns(a=0, b=0),
        guards={
            "G_N2c_3_cap0_same_source": {"status": "conflict", "conflicts": ["eval_set_sha256"]},
            "G_N2c_3_replay_same_source": {"status": "ok", "conflicts": []},
            "G_N2c_4_disjointness": measured_separation(),
        },
    )
    loaded = c4.load_pair(conflict)
    assert c4.comparability_judgement(loaded)["passed"] is False
    verdict = c4.adjudicate(loaded, c4.load_gain(args[0]), c4.floor_quality(args[1]), ruler)
    assert verdict["status"] == "cost_persists" or verdict["criteria"]["J_N02_3_comparability"][
        "conflicts"
    ] == ["eval_set_sha256"], verdict


def test_a_swapped_face_or_a_missing_gain_field_refuses(tmp_path: Path) -> None:
    pair = make_pair(tmp_path, columns(a=0, b=0))
    path = Path(json.loads(pair.read_text(encoding="utf-8"))["cap0_after_path"])
    path.write_text('{"identity": "tampered"}', encoding="utf-8", newline="\n")
    with pytest.raises(c4.Refuse, match="bytes changed under the pair"):
        c4.load_pair(pair)

    empty = write(tmp_path / "empty_gain.json", {})
    with pytest.raises(c4.Refuse, match=c4.GAIN_QUALITY_KEY):
        c4.load_gain(empty)

    wrong = write(tmp_path / "wrong.json", {"format": "other", "prereg": "x", "per_column": {}})
    with pytest.raises(c4.Refuse, match="retention pair must be"):
        c4.load_pair(wrong)


def test_the_separation_guard_passes_a_known_good_and_fails_two_negatives(tmp_path: Path) -> None:
    """新写的守卫必须先跑一次"已知-good"：这条用的是真封存件的词汇，两条负对照各打断一种成功条件。"""
    real = c4.load_pair(REPO / "reports/taiji_n2_04_retention_pair_n2c1_20261009.json")
    block = c4.separation_judgement(real)
    assert block["passed"] is True, block
    assert (block["overlap"], block["state"], block["windows"]) == (0, "measured", 116), block

    dirty = c4.separation_judgement(
        {"guards": {"G_N2c_4_disjointness": measured_separation(overlap=3)}}
    )
    assert dirty["passed"] is False and dirty["overlap"] == 3, dirty

    missing = c4.separation_judgement(
        {"guards": {"G_N2c_4_disjointness": {"status": "unverified", "reason": "no manifest"}}}
    )
    assert missing["passed"] is False, missing
    ruler = {"state": "present", "checked_face": "x", "key": c4.RANGE_KEY, "value": 7}
    verdict = c4.adjudicate(
        c4.load_pair(
            make_pair(
                tmp_path,
                columns(a=0, b=0),
                guards={
                    "G_N2c_3_cap0_same_source": {"status": "ok", "conflicts": []},
                    "G_N2c_3_replay_same_source": {"status": "ok", "conflicts": []},
                    "G_N2c_4_disjointness": {"status": "unverified"},
                },
            )
        ),
        c4.load_gain(make_gain(tmp_path, 9.0)),
        c4.floor_quality(make_floor(tmp_path, 1.0)),
        ruler,
    )
    assert verdict["status"] == "separation_unverified", verdict["status"]


def test_cli_writes_once_and_reports_the_real_gap(tmp_path: Path) -> None:
    """真实封存件那一支：今天的合法产物就是"缺哪一列"的清单，而不是任何能力主张。"""
    out = tmp_path / "c4.json"
    rc = c4.main(
        [
            "--retention-pair",
            str(REPO / "reports/taiji_n2_04_retention_pair_n2c1_20261009.json"),
            "--gain",
            str(REPO / "reports/taiji_n2b_powerup_20261008_n2c1.json"),
            "--noise-floor",
            str(REPO / "reports/taiji_n5_10_noise_floor_retention_pair_20261011.json"),
            "--out-report",
            str(out),
        ]
    )
    assert rc == 1, rc
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "range_unverified", payload["status"]
    assert payload["models_loaded"] == 0 and payload["parameter_updates"] == 0
    assert payload["is_capability_claim"] is False
    assert payload["criteria"]["J_N02_2_retention"]["verdict"] == "cost_persists"
    assert payload["criteria"]["J_N02_1_gain"]["noise_floor"] is None
    assert payload["criteria"]["J_N02_4_separation"]["passed"] is True, payload["criteria"]
    assert payload["criteria"]["J_N02_3_comparability"]["passed"] is True
    before = out.read_bytes()
    assert (
        c4.main(
            [
                "--retention-pair",
                str(REPO / "reports/taiji_n2_04_retention_pair_n2c1_20261009.json"),
                "--gain",
                str(REPO / "reports/taiji_n2b_powerup_20261008_n2c1.json"),
                "--out-report",
                str(out),
            ]
        )
        == 2
    )
    assert out.read_bytes() == before, "拒绝不得改写已封存的件"
    assert c4.main(["--retention-pair", "x", "--gain", "y", "--out-report", str(out)]) == 2
