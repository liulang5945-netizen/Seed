"""PLAN-N5-08 的 `J-N5b-5` 取数式契约测（`adjudicate_taiji_n5_shadow_gate.py`）。

这一格从前是硬编码字面量（DEBT-G89），所以每条分支都要能红、也要能绿：
地板列剔除（G-N5g-2）、噪声档前置（J-N5g-2）、同源守卫（G-N5g-1）、两臂不一致（J-N5g-3）、列数守卫。
「不给旗标时逐位不变」是兼容锚——已入库的 ㊵-659 读数不许被本笔追认或改写。
真件那支读的是已封存的 `reports/taiji_n5_07_*_pair_20261010.json`：夹具是自造的，证不了取数键；
我预注册里恰好把键写错过一次（把 `criteria.columns_compared` 当成列名列表），只有真件能证。
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n5_shadow_gate.py"
PAIR_FORMAT = "taiji-n2-04-retention-pair-v1"
TREATED_PAIR = REPO / "reports" / "taiji_n5_07_treated_pair_20261010.json"
CONTROL_PAIR = REPO / "reports" / "taiji_n5_07_control_pair_20261010.json"
NOISE_FLOOR_PAIR = REPO / "reports" / "taiji_n5_08_determinism_pair_20261010.json"

COLUMNS = (
    "cap0:E",
    "replay_well_formed:0.0",
    "replay_well_formed:0.5",
    "replay_well_formed:1.0",
    "replay_well_formed:2.0",
    "replay_strict_hits:1.0",
    "replay_strict_hits:2.0",
)
#: 基线取 ㊵-656 现读的 3／34／51／54／60，两臂 after 取那张表的治疗列；`strict_hits` 两档基线即 0＝地板。
TREATED = {
    "cap0:E": (3, 2),
    "replay_well_formed:0.0": (34, 19),
    "replay_well_formed:0.5": (51, 32),
    "replay_well_formed:1.0": (54, 34),
    "replay_well_formed:2.0": (60, 36),
    "replay_strict_hits:1.0": (0, 0),
    "replay_strict_hits:2.0": (0, 0),
}
CONTROL = {
    "cap0:E": (3, 1),
    "replay_well_formed:0.0": (34, 15),
    "replay_well_formed:0.5": (51, 35),
    "replay_well_formed:1.0": (54, 42),
    "replay_well_formed:2.0": (60, 32),
    "replay_strict_hits:1.0": (0, 0),
    "replay_strict_hits:2.0": (0, 0),
}
#: 逐列差 = control.after − treated.after（本笔手推，与真件那支的仪器读数互为对照）。
DROPS = {"cap0:E": -1, "replay_well_formed:0.0": -4, "replay_well_formed:0.5": 3, "replay_well_formed:1.0": 8, "replay_well_formed:2.0": -4}


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_gate_judge_retention", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = _load()


def _pair(
    tmp_path: Path,
    name: str,
    cells: dict[str, tuple[int, int]],
    *,
    cap0_before: str = r"reports\base_beta_cap0.json",
    replay_before: str = r"reports\base_beta_replay24.json",
    cap0_sha: str = "aa11",
    replay_sha: str = "bb22",
    columns_compared: Any = 7,
    verdict: str = "cost_persists",
    cap0_guard: str = "ok",
    replay_guard: str = "ok",
) -> Path:
    payload: dict[str, Any] = {
        "format": PAIR_FORMAT,
        "status": "ok",
        "verdict": verdict,
        "cap0_before_path": cap0_before,
        "replay_before_path": replay_before,
        "cap0_before_sha256": cap0_sha,
        "replay_before_sha256": replay_sha,
        "criteria": {"columns_compared": columns_compared},
        "guards": {
            "G_N2c_3_cap0_same_source": {"status": cap0_guard},
            "G_N2c_3_replay_same_source": {"status": replay_guard},
        },
        "per_column": {
            column: {"before": before, "after": after} for column, (before, after) in cells.items()
        },
    }
    path = tmp_path / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def _both(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _pair(tmp_path, "treated_pair.json", TREATED),
        _pair(tmp_path, "control_pair.json", CONTROL),
    )


def _replicate(tmp_path: Path, offsets: dict[str, int]) -> Path:
    cells = {
        column: (CONTROL[column][0], CONTROL[column][1] + offsets.get(column, 0))
        for column in COLUMNS
    }
    return _pair(tmp_path, "replicate_pair.json", cells)


def _block(gate: float) -> dict[str, Any]:
    return {
        "candidate_id": "r4-candidate:abc",
        "bridge_id": "predictive_residual.bridge",
        "gate": gate,
        "unit_count": 97,
        "shadow_gate_requested": gate,
        "candidate_gate": 0.5,
        "candidate_utility": 0.31 if gate else 0.0,
        "candidate_counterfactual_utility": 0.11 if gate else 0.0,
    }


def _arms(tmp_path: Path) -> tuple[str, str]:
    paths = []
    for side, gate in (("treated", 1.0), ("control", 0.0)):
        path = tmp_path / f"{side}.pt"
        torch.save({"envelope": {"format": "seed-native-v1", "n5_shadow": _block(gate)}}, str(path))
        paths.append(str(path))
    return paths[0], paths[1]


def _run(tmp_path: Path, extra: list[str]) -> tuple[int, dict[str, Any]]:
    treated, control = _arms(tmp_path)
    out = tmp_path / "verdict.json"
    rc = JUDGE.main(
        ["--arm-treated", treated, "--arm-control", control, "--out", str(out)] + extra
    )
    return rc, json.loads(out.read_text(encoding="utf-8"))


def _retention_args(treated: Path, control: Path, replicate: Path | None = None) -> list[str]:
    argv = ["--retention-treated", str(treated), "--retention-control", str(control)]
    if replicate is not None:
        argv += ["--noise-floor-arm", str(replicate)]
    return argv


def test_no_retention_flags_leaves_the_placeholder_and_publishes_no_lane_block(
    tmp_path: Path,
) -> None:
    #: 兼容锚（㊵-665 起按 PLAN-N5-09 G-N5h-4 的新形状钉）：不给三枚旗标时**那一格键**仍是原话、
    #: 且不出 `retention_lane` 块 ⇒ ㊵-659 那份已封存读数不被改写。
    #: 原句「整个 payload 逐字节相同」随 PLAN-N5-09 作废并**换成键级三条**——合取接上取数式后
    #: `j_n5b_6` 与新增的 `conjunction` 必然出现在输出里，那正是 PLAN-N5-09 的目的；
    #: 本锚保护的对象从来是"已入库读数不被追认"，键级断言同样保护得住（另有 sha 钉死在 n5_19）。
    rc, payload = _run(tmp_path, [])
    assert payload["j_n5b_5"] == "unverified_retention_lane_not_run"
    assert "retention_lane" not in payload
    assert payload["j_n5b_6"] == "unverified"
    #: 重推过的期望（不是抄仪器输出）：这一支 fixture 里 3 无面⇒未测、4 无件⇒未测、5 无旗标⇒未测，
    #: 只有 2 成立；`missing` 必空是 G-N5h-3（未判与不成立不许同句共现）。
    assert payload["conjunction"]["unmeasured"] == ["J-N5b-3", "J-N5b-4", "J-N5b-5"]
    assert payload["conjunction"]["missing"] == []
    assert payload["conjunction"]["branches"]["j_n5b_2"]["state"] == "established"
    assert rc == 0


def test_real_pair_reports_read_as_noise_floor_missing_with_two_floor_columns(tmp_path: Path) -> None:
    #: 真件那支：列名、地板列、逐列差全部由仪器算出，期望值是本笔之前手推并已入库的那张表。
    #: 这一条同时钉住取数键——列名住在 `per_column`，`criteria.columns_compared` 是计数。
    rc, payload = _run(tmp_path, _retention_args(TREATED_PAIR, CONTROL_PAIR))
    lane = payload["retention_lane"]
    assert payload["j_n5b_5"] == "unverified_noise_floor_missing"
    assert lane["columns_compared"] == 7
    assert sorted(lane["not_judgable_floor"]) == [
        "replay_strict_hits:1.0",
        "replay_strict_hits:2.0",
    ]
    assert set(lane["not_judgable_floor"].values()) == {0}
    assert lane["drop_control_minus_treated"] == DROPS
    assert lane["judgable_columns"] == sorted(DROPS)
    #: §7 发表资格前置③：两臂 verdict 必须一起出版。
    assert lane["arm_verdicts"] == {"treated": "cost_persists", "control": "cost_persists"}
    assert rc == 0


def test_real_three_arm_read_publishes_cost_persists_at_zero_noise_floor(tmp_path: Path) -> None:
    #: ㊵-663 的头牌锚：第三档（同码同参、不带 `--n5-shadow-gate`）出版的 `noise_floor` 现读 **0**，
    #: ⇒ 五列全部 resolved、`j_n5b_5 = cost_persists`，且"治疗臂比对照臂在两列上更差 3 与 8"**不再**
    #: 能被解释成浮点非确定性（那条解释由 `CONTROL_VS_DETERMINISM = 0 of 220` 逐张量读数排除）。
    #: 期望值是本笔之前手推并已入库的那张差值表，不是从仪器输出抄回来的。
    rc, payload = _run(tmp_path, _retention_args(TREATED_PAIR, CONTROL_PAIR, NOISE_FLOOR_PAIR))
    lane = payload["retention_lane"]
    assert payload["j_n5b_5"] == "cost_persists"
    assert lane["noise_floor"] == 0
    assert lane["resolved_columns"] == sorted(DROPS)
    assert lane["not_resolved"] == []
    assert lane["drop_control_minus_treated"] == DROPS
    assert lane["noise_floor_pair"].endswith("taiji_n5_08_determinism_pair_20261010.json")
    assert rc == 0


def test_synthetic_fixture_reproduces_the_real_column_arithmetic(tmp_path: Path) -> None:
    #: 夹具与真件互为对照：同一条取数式在两份来源上必须给同一张差值表，否则夹具在自证自己。
    treated, control = _both(tmp_path)
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["retention_lane"]["drop_control_minus_treated"] == DROPS
    assert rc == 0


def test_same_basename_different_bytes_is_not_same_source(tmp_path: Path) -> None:
    #: 「同源」从前比的是文件名 ⇒ 两臂各自目录里同名不同字节的两张面会假过。改成比摘要后必须红。
    treated = _pair(tmp_path, "treated_pair.json", TREATED)
    control = _pair(
        tmp_path,
        "control_pair.json",
        CONTROL,
        cap0_before=r"output\arm_b\base_beta_cap0.json",
        cap0_sha="cc33",
    )
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["j_n5b_5"] == "faces_not_same_source"
    faces = payload["retention_lane"]["baseline_faces"]
    assert faces["treated"]["cap0_before"] == faces["control"]["cap0_before"]
    assert faces["treated"]["cap0_before_sha256"] == "aa11"
    assert faces["control"]["cap0_before_sha256"] == "cc33"
    assert rc == 2


def test_declared_column_count_must_match_the_column_keys(tmp_path: Path) -> None:
    treated = _pair(tmp_path, "treated_pair.json", TREATED, columns_compared=8)
    control = _pair(tmp_path, "control_pair.json", CONTROL, columns_compared=8)
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["j_n5b_5"] == "column_count_mismatch"
    assert payload["retention_lane"] == {"declared": 8, "found": 7}
    assert rc == 2


def test_column_count_as_a_list_is_rejected_not_iterated(tmp_path: Path) -> None:
    #: 负对照：预注册里我误把它当列表。今天必须是「对不上就响亮退出」，不能被 `sorted()` 蒙混。
    treated = _pair(tmp_path, "treated_pair.json", TREATED, columns_compared=list(COLUMNS))
    control = _pair(tmp_path, "control_pair.json", CONTROL, columns_compared=list(COLUMNS))
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["j_n5b_5"] == "column_count_mismatch"
    assert payload["retention_lane"]["found"] == 7
    assert rc == 2


def test_control_arm_columns_have_to_be_the_same_set(tmp_path: Path) -> None:
    treated = _pair(tmp_path, "treated_pair.json", TREATED)
    control = _pair(
        tmp_path, "control_pair.json", {k: v for k, v in CONTROL.items() if k != "cap0:E"},
        columns_compared=6,
    )
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["j_n5b_5"] == "column_sets_differ"
    assert rc == 2


def test_a_single_differing_baseline_voids_the_pair(tmp_path: Path) -> None:
    treated = _pair(tmp_path, "treated_pair.json", TREATED)
    shifted = dict(CONTROL)
    shifted["cap0:E"] = (4, 1)
    control = _pair(tmp_path, "control_pair.json", shifted)
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["j_n5b_5"] == "baseline_differs"
    assert payload["retention_lane"]["column"] == "cap0:E"
    assert rc == 2


def test_dropping_more_than_four_columns_stops_before_any_value(tmp_path: Path) -> None:
    #: 地板列被剔后 `judgable < 3` ⇒ 整枚不判，不拿剩下的两列出方向。
    cells = {column: (0, 5) for column in COLUMNS}
    treated = _pair(tmp_path, "treated_pair.json", cells)
    control = _pair(tmp_path, "control_pair.json", cells)
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["j_n5b_5"] == "not_judgable_below_min"
    assert len(payload["retention_lane"]["not_judgable_floor"]) == 7
    assert rc == 0


def test_noise_floor_that_still_resolves_four_columns_publishes_the_cost(tmp_path: Path) -> None:
    treated, control = _both(tmp_path)
    replicate = _replicate(tmp_path, {"cap0:E": 2})
    rc, payload = _run(tmp_path, _retention_args(treated, control, replicate))
    lane = payload["retention_lane"]
    assert lane["noise_floor"] == 2
    assert lane["resolved_columns"] == [
        "replay_well_formed:0.0",
        "replay_well_formed:0.5",
        "replay_well_formed:1.0",
        "replay_well_formed:2.0",
    ]
    assert lane["not_resolved"] == ["cap0:E"]
    assert payload["j_n5b_5"] == "cost_persists"
    assert rc == 0


def test_a_noise_floor_as_large_as_the_smallest_drop_leaves_no_range_to_publish(tmp_path: Path) -> None:
    #: 出值前置的第二支：被噪声吞掉之后不足 3 列 ⇒ 整枚不判，且不写「保持成立」。
    treated, control = _both(tmp_path)
    replicate = _replicate(tmp_path, {"cap0:E": 4, "replay_well_formed:0.0": -4})
    rc, payload = _run(tmp_path, _retention_args(treated, control, replicate))
    lane = payload["retention_lane"]
    assert lane["noise_floor"] == 4
    assert lane["not_resolved"] == [
        "cap0:E",
        "replay_well_formed:0.0",
        "replay_well_formed:0.5",
        "replay_well_formed:2.0",
    ]
    assert payload["j_n5b_5"] == "not_resolved_insufficient_range"
    assert "resolved_columns" not in lane
    assert rc == 0


def test_zero_noise_floor_and_no_upward_drop_is_the_holds_branch(tmp_path: Path) -> None:
    #: 这一支必须能为真：只有占位符时它永远出不来。逐列差全 ≤ 0（治疗臂每列都不低于控制臂）
    #: 且第三档逐位复现控制臂 ⇒ `noise_floor = 0`、五列全部 resolved、`holds`。
    cells_t = {
        "cap0:E": (3, 3),
        "replay_well_formed:0.0": (34, 34),
        "replay_well_formed:0.5": (51, 51),
        "replay_well_formed:1.0": (54, 54),
        "replay_well_formed:2.0": (60, 60),
        "replay_strict_hits:1.0": (0, 0),
        "replay_strict_hits:2.0": (0, 0),
    }
    treated = _pair(tmp_path, "treated_pair.json", cells_t)
    control = _pair(tmp_path, "control_pair.json", CONTROL)
    replicate = _pair(tmp_path, "replicate_pair.json", CONTROL)
    rc, payload = _run(tmp_path, _retention_args(treated, control, replicate))
    lane = payload["retention_lane"]
    assert lane["noise_floor"] == 0
    assert payload["j_n5b_5"] == "holds"
    assert lane["resolved_columns"] == sorted(DROPS)
    assert lane["not_resolved"] == []
    assert rc == 0


def test_two_arms_with_different_verdicts_are_arm_dependent_not_the_lighter_one(tmp_path: Path) -> None:
    #: J-N5g-3：不许取对自己有利的那一臂。
    treated = _pair(tmp_path, "treated_pair.json", TREATED, verdict="retention_holds")
    control = _pair(tmp_path, "control_pair.json", CONTROL, verdict="cost_persists")
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["j_n5b_5"] == "arm_dependent"
    assert payload["retention_lane"]["arm_verdicts"] == {
        "treated": "retention_holds",
        "control": "cost_persists",
    }
    assert rc == 0


def test_a_pair_report_whose_own_same_source_guard_is_not_ok_voids_the_lane(tmp_path: Path) -> None:
    #: G-N5g-1：本器不重造同源判定，只读配对件出版的结论；非 `ok` ⇒ 未判＋rc=2。
    treated = _pair(tmp_path, "treated_pair.json", TREATED, replay_guard="conflict")
    control = _pair(tmp_path, "control_pair.json", CONTROL)
    rc, payload = _run(tmp_path, _retention_args(treated, control))
    assert payload["j_n5b_5"] == "unverified_same_source_guard_not_ok"
    assert payload["retention_lane"]["guard_failures"] == ["treated.G_N2c_3_replay_same_source"]
    assert rc == 2


def test_noise_floor_arm_on_another_base_is_refused(tmp_path: Path) -> None:
    treated, control = _both(tmp_path)
    replicate = _pair(tmp_path, "replicate_pair.json", CONTROL, replay_sha="zz99")
    rc, payload = _run(tmp_path, _retention_args(treated, control, replicate))
    assert payload["j_n5b_5"] == "faces_not_same_source"
    assert payload["retention_lane"]["noise_floor_pair"].endswith("replicate_pair.json")
    assert rc == 2


def test_noise_floor_arm_missing_a_column_is_loud(tmp_path: Path) -> None:
    treated, control = _both(tmp_path)
    cells = {column: CONTROL[column] for column in COLUMNS if column != "cap0:E"}
    replicate = _pair(tmp_path, "replicate_pair.json", cells, columns_compared=6)
    rc, payload = _run(tmp_path, _retention_args(treated, control, replicate))
    assert payload["j_n5b_5"] == "noise_floor_columns_incomplete"
    assert rc == 2


def test_reader_refuses_a_foreign_format_and_a_missing_key(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.json"
    foreign.write_text(
        json.dumps({"format": "taiji-n3-pressure-face-v2"}) + "\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match="format"):
        JUDGE._read_pair(foreign)
    payload = json.loads(TREATED_PAIR.read_text(encoding="utf-8"))
    del payload["cap0_before_sha256"]
    no_sha = tmp_path / "no_sha.json"
    no_sha.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match="cap0_before_sha256"):
        JUDGE._read_pair(no_sha)


def test_retention_lane_carries_no_threshold_literal_and_the_scanner_can_see_one() -> None:
    #: G-N5g-3（零假定）：`noise_floor` 只能来自输入件，本函数不许自带阈值。
    #: 允许的整数只有预注册里写死的那两个计数（`judgable ≥ 3` 的 3、地板定义的 0）与布尔语义的 1/2。
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    numbers = [
        node.value
        for node in ast.walk(functions["judge_retention_lane"])
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]
    assert not [n for n in numbers if isinstance(n, float)], numbers
    assert set(numbers) <= {0, 1, 2, 3}, numbers
    #: 同一台扫描器在 `main` 上必须能抓到东西 ⇒ 否则上面两条是空扫（0.02 是 PLAN-N5-02/03 冻的过线界，
    #: 住在那里的旗标默认值上，本笔不搬——已按 DEBT-G90 登记）。
    main_numbers = [
        node.value
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    ]
    assert 0.02 in main_numbers, main_numbers
