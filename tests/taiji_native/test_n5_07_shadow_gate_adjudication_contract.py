"""N5 通电判读器（`adjudicate_taiji_n5_shadow_gate.py`）的契约测。

PLAN-N5-02 J-N5b-1／J-N5b-2 的每一支都必须能为假，且 fail-closed 那一支要能在真件上走出来：
本仓已被「缺列被读成满足」绊过（㊵-590 的 `get()` 假 null），所以这里的正负例都直接构造检查点。

夹具全部落 `tmp_path`（`torch.save` 一个含 `envelope` 的 dict），不碰 `output/` 与 `reports/`。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n5_shadow_gate.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_gate_judge", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = _load()


def _face(tmp_path: Path, name: str, block: dict[str, Any] | None) -> Path:
    path = tmp_path / name
    envelope: dict[str, Any] = {"format": "seed-native-v1"}
    if block is not None:
        envelope["n5_shadow"] = block
    torch.save({"envelope": envelope}, str(path))
    return path


def _full(**over: Any) -> dict[str, Any]:
    block = {
        "candidate_id": "r4-candidate:abc",
        "bridge_id": "predictive_residual.bridge",
        "gate": 1.0,
        "unit_count": 97,
        "shadow_gate_requested": 1.0,
        "candidate_gate": 0.5,
        "candidate_utility": 0.31,
        "candidate_counterfactual_utility": 0.11,
    }
    block.update(over)
    return block


def _run(tmp_path: Path, treated: Path, control: Path | None = None) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "verdict.json"
    argv = ["--arm-treated", str(treated), "--out", str(out)]
    if control is not None:
        argv += ["--arm-control", str(control)]
    rc = JUDGE.main(argv)
    import json

    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_learning_branch_is_learned(tmp_path: Path) -> None:
    rc, payload = _run(tmp_path, _face(tmp_path, "t.pt", _full()))
    assert rc == 0
    arm = payload["arms"]["treated"]
    assert arm["j_n5b_1"] == "present"
    assert arm["j_n5b_2"] == "shadow_learned"


def test_powered_but_zero_utilities_is_inert(tmp_path: Path) -> None:
    face = _face(
        tmp_path,
        "t.pt",
        _full(candidate_utility=0.0, candidate_counterfactual_utility=0.0),
    )
    rc, payload = _run(tmp_path, face)
    assert rc == 1
    assert payload["arms"]["treated"]["j_n5b_2"] == "shadow_inert"


def test_unpowered_is_not_powered_not_inert(tmp_path: Path) -> None:
    #: F 跑的形状：物化并挂上，但 `_gate` 仍 0 ⇒ 归 `not_powered`，不混进 `shadow_inert`。
    rc, payload = _run(tmp_path, _face(tmp_path, "t.pt", _full(gate=0.0)))
    assert rc == 1
    assert payload["arms"]["treated"]["j_n5b_2"] == "not_powered"


def test_missing_self_report_key_is_fail_closed(tmp_path: Path) -> None:
    block = _full()
    del block["candidate_counterfactual_utility"]
    rc, payload = _run(tmp_path, _face(tmp_path, "t.pt", block))
    arm = payload["arms"]["treated"]
    assert rc == 2
    assert arm["j_n5b_1"] == "ran_not_measured"
    assert arm["missing_self_report_keys"] == ["candidate_counterfactual_utility"]
    assert arm["j_n5b_2"] == "unverified_missing_face"


def test_block_absent_entirely_reports_all_five_missing(tmp_path: Path) -> None:
    rc, payload = _run(tmp_path, _face(tmp_path, "t.pt", None))
    arm = payload["arms"]["treated"]
    assert rc == 2
    assert len(arm["missing_self_report_keys"]) == 5


def test_two_arms_with_identical_requested_gate_are_unpairable(tmp_path: Path) -> None:
    treated = _face(tmp_path, "t.pt", _full())
    control = _face(tmp_path, "c.pt", _full())
    rc, payload = _run(tmp_path, treated, control)
    assert rc == 2
    assert payload["pairing"]["status"] == "pairing_invalid"


def test_missing_face_file_is_rejected(tmp_path: Path) -> None:
    #: `main()` 返回码就是判据（`__main__` 才包 `raise SystemExit`），所以这里读 rc 不读异常。
    rc = JUDGE.main(["--arm-treated", str(tmp_path / "nope.pt"), "--out", str(tmp_path / "v.json")])
    assert rc == 2


def _pressure_face(tmp_path: Path, name: str, **over: Any) -> Path:
    import json

    header = {
        "kind": "face",
        "format": "taiji-n3-pressure-face-v2",
        "seed": 20260822,
        "readout": "predictive",
        "policy": {"minimum_pressure": 0.65, "required_pressure_steps": 3},
        "developmental": {
            "fast_slow_requested": True,
            "bridge_gate_requested": 1.0,
            "bridge_gate_actual": 1.0,
        },
    }
    for key, value in over.items():
        if key == "policy.minimum_pressure":
            header["policy"]["minimum_pressure"] = value
        elif key == "developmental.fast_slow_requested":
            header["developmental"]["fast_slow_requested"] = value
        else:
            header[key] = value
    path = tmp_path / name
    lines = [json.dumps(header, ensure_ascii=False)]
    lines.append(json.dumps({"kind": "pressure", "decision_should_propose": False}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _pair(tmp_path: Path, treated_gate: float, control_gate: float) -> tuple[Path, Path]:
    return (
        _face(tmp_path, "t.pt", _full(shadow_gate_requested=treated_gate, gate=treated_gate)),
        _face(tmp_path, "c.pt", _full(shadow_gate_requested=control_gate, gate=control_gate)),
    )


def test_quadruple_identical_faces_make_the_pair_valid(tmp_path: Path) -> None:
    treated, control = _pair(tmp_path, 1.0, 0.0)
    face_t = _pressure_face(tmp_path, "tp.jsonl")
    face_c = _pressure_face(tmp_path, "cp.jsonl")
    out = tmp_path / "v.json"
    rc = JUDGE.main(
        [
            "--arm-treated",
            str(treated),
            "--arm-control",
            str(control),
            "--face-treated",
            str(face_t),
            "--face-control",
            str(face_c),
            "--out",
            str(out),
        ]
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pairing"]["status"] == "pairing_valid_on_quadruple"
    assert payload["pairing"]["j_n5b_3"] == "valid"
    #: 对照臂不通电 ⇒ 它自己该被判 `not_powered`，但那是它的判据，不是配对的失败。
    assert payload["arms"]["control"]["j_n5b_2"] == "not_powered"
    #: 对照臂本该不通电，所以 rc 不由它抬——它验的是配对有效性。
    assert rc == 0


def test_a_single_differing_quadruple_dimension_invalidates_the_pair(tmp_path: Path) -> None:
    treated, control = _pair(tmp_path, 1.0, 0.0)
    face_t = _pressure_face(tmp_path, "tp.jsonl")
    face_c = _pressure_face(tmp_path, "cp.jsonl", **{"policy.minimum_pressure": 0.7})
    out = tmp_path / "v.json"
    rc = JUDGE.main(
        [
            "--arm-treated",
            str(treated),
            "--arm-control",
            str(control),
            "--face-treated",
            str(face_t),
            "--face-control",
            str(face_c),
            "--out",
            str(out),
        ]
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pairing"]["status"] == "pairing_invalid"
    assert payload["pairing"]["reason"] == "quadruple_differs"
    assert payload["pairing"]["differing"] == ["tau"]
    assert rc == 2


def test_faces_absent_leaves_the_quadruple_unverified_not_valid(tmp_path: Path) -> None:
    #: 关键的一支：只给两枚检查点时，配对**不许**被读成成立（缺现场四元组＝未判）。
    treated, control = _pair(tmp_path, 1.0, 0.0)
    out = tmp_path / "v.json"
    rc = JUDGE.main(
        [
            "--arm-treated",
            str(treated),
            "--arm-control",
            str(control),
            "--out",
            str(out),
        ]
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pairing"]["status"] == "gate_differs_quadruple_unverified"
    assert payload["pairing"]["j_n5b_3"] == "unverified"
    assert rc == 0


def _metric_file(
    tmp_path: Path,
    treated_means: list[float],
    control_means: list[float],
    complete_treated: bool = True,
    band_treated: float = 0.01,
    band_control: float = 0.01,
) -> Path:
    import json as _json

    path = tmp_path / "n3a.json"
    path.write_text(
        _json.dumps(
            {
                "arms": {
                    "arm_A": {
                        "acc_segment_means": treated_means,
                        "measurement_complete": complete_treated,
                        "acc_noise_band_adjacent_max": band_treated,
                    },
                    "arm_B": {
                        "acc_segment_means": control_means,
                        "measurement_complete": True,
                        "acc_noise_band_adjacent_max": band_control,
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _pair_args(tmp_path: Path, extra: list[str]) -> list[str]:
    treated, control = _pair(tmp_path, 1.0, 0.0)
    out = tmp_path / "v.json"
    argv = ["--arm-treated", str(treated), "--arm-control", str(control), "--out", str(out)]
    return argv, out


def test_metric_lane_arithmetic_is_mechanical(tmp_path: Path) -> None:
    #: 期望值是**事先手推的小整数**，不是抄仪器输出：0.30/0.10 ⇒ treated 末两段 0.20、
    #: control 末两段 0.05 ⇒ delta 0.15 > 0.02 ⇒ exceeds_line 为真。
    metric = _metric_file(tmp_path, [0.10, 0.30], [0.00, 0.10])
    argv, out = _pair_args(tmp_path, [])
    rc = JUDGE.main(argv + ["--n3a-pair-file", str(metric), "--line", "0.02"])
    payload = json.loads(out.read_text(encoding="utf-8"))
    lane = payload["metric_lane"]
    assert lane["last_two_segment_mean"] == {"treated": 0.20, "control": 0.05}
    assert abs(lane["delta_treated_minus_control"] - 0.15) < 1e-12
    assert lane["exceeds_line"] is True
    assert rc == 0


def test_metric_lane_is_negative_when_treated_is_lower(tmp_path: Path) -> None:
    #: 反号一支必须能为假：treated 比 control 低时 exceeds_line 必为 False，delta 为负。
    metric = _metric_file(tmp_path, [0.10, 0.10], [0.30, 0.30])
    argv, out = _pair_args(tmp_path, [])
    JUDGE.main(argv + ["--n3a-pair-file", str(metric)])
    lane = json.loads(out.read_text(encoding="utf-8"))["metric_lane"]
    assert lane["delta_treated_minus_control"] < 0.0
    assert lane["exceeds_line"] is False


def test_incomplete_metric_side_is_never_read_as_a_verdict(tmp_path: Path) -> None:
    metric = _metric_file(tmp_path, [0.10, 0.30], [0.00, 0.10], complete_treated=False)
    argv, out = _pair_args(tmp_path, [])
    JUDGE.main(argv + ["--n3a-pair-file", str(metric)])
    lane = json.loads(out.read_text(encoding="utf-8"))["metric_lane"]
    assert lane["j_n5b_4"] == "unverified_metric_incomplete"
    assert "delta_treated_minus_control" not in lane


def test_missing_metric_file_keeps_the_conjunct_unverified(tmp_path: Path) -> None:
    argv, out = _pair_args(tmp_path, [])
    JUDGE.main(argv)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metric_lane"]["j_n5b_4"] == "unverified_no_metric_file"
    #: 〔㊵-665 期望值随 PLAN-N5-09 更新，**不是放宽**：这一格从前只钉一句字面量
    #: `not_adjudicable_until_2_3_4_5_are_all_measured`；接上取数式后同一输入必须出 `unverified`
    #: **并点名**缺的是哪一支。原断言"合取不许在缺任一支时给出『有贡献』"完整保留，且新增了两条更硬的：
    #: 值必须在 PLAN-N5-09 §3 的三形之内、`unmeasured` 数组必须含 J-N5b-4。
    assert payload["j_n5b_6"] == "unverified"
    assert payload["j_n5b_6"] != "established"
    assert "J-N5b-4" in payload["conjunction"]["unmeasured"]
    assert payload["conjunction"]["missing"] == []


def test_band_swallowing_the_line_voids_the_comparison(tmp_path: Path) -> None:
    #: PLAN-N5-03 §2 第一支：过线界被噪声带吞掉 ⇒ 比较作废，既不写成立也不写不成立，
    #: 且 rc 必须抬起（绿色判读不等于判完了）。delta 很大也一样作废。
    metric = _metric_file(
        tmp_path, [0.10, 0.90], [0.00, 0.10], band_treated=0.05, band_control=0.04
    )
    argv, out = _pair_args(tmp_path, [])
    rc = JUDGE.main(argv + ["--n3a-pair-file", str(metric), "--line", "0.02"])
    payload = json.loads(out.read_text(encoding="utf-8"))
    lane = payload["metric_lane"]
    assert lane["ruler_usable"] is False
    assert lane["j_n5b_4"] == "ruler_unusable"
    assert sorted(lane["swallowed_by"]) == ["control", "treated"]
    assert "exceeds_line" not in lane
    assert rc == 1


def test_usable_ruler_publishes_verdict_and_resolution_note(tmp_path: Path) -> None:
    #: 手推夹具：delta 0.001 既不过线也落在带内 ⇒ `not_holds` ＋"分辨不了这个量级"的提示。
    #: 这条提示是**仪器分辨率陈述**，不是"影子无效应"——两者不许互换。
    metric = _metric_file(tmp_path, [0.100, 0.102], [0.100, 0.100])
    argv, out = _pair_args(tmp_path, [])
    rc = JUDGE.main(argv + ["--n3a-pair-file", str(metric), "--line", "0.02"])
    lane = json.loads(out.read_text(encoding="utf-8"))["metric_lane"]
    assert lane["ruler_usable"] is True
    assert lane["verdict"] == "not_holds"
    assert lane["j_n5b_4"] == "not_holds"
    assert lane["within_noise_band"] is True
    assert rc == 0
