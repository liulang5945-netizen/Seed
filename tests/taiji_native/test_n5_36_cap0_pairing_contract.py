"""`PLAN-N5-22` 的 J-N5u-1／2／3 载体（`audit_taiji_n5_22_cap0_pairing.py`）的契约测。

这台仪器的存在理由是"**资格先于方向**"：两份 CAP 件只要评测集 sha／HEAD／链路／计分纪律任何一处不齐，
它必须拒绝给出方向（`pairing_invalid`），而不是照样减一减。另一半是"没测到的题不许当 0 分"——
`pending_human_review` 或 `load_ok` 不为真的题必须落在 `excluded_ids` 里带原因出版，
且不进翻转计数（[[probe-output-must-be-verified-present]]）。

测面全部是合成两份件（真长跑产物不进测，不可复现），形状照已入库件
`reports/taiji_n5_07_base_beta_cap0_20261010.json` 逐字段键名搭：
`identity.{git_head,checkpoint_sha256,eval_set_sha256}`、`chain`、`trained_during_eval`、
`scoring_discipline`、`dimensions[code].{item_count,error,tally,items[]}`，题里是 `id`／`score`／
`load_ok`／`pending_human_review`。分支互斥也要能为真：一维 +3 一维 −4 时必须落 `direction_inconsistent`，
不许挑一支写成增益。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_n5_22_cap0_pairing.py"
DISCIPLINE = [
    "固定答案题精确计分",
    "自动评审只辅助",
    "模板变量不算独立样本",
    "测试时不训练",
    "未测记 untested",
]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_22_pairing", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CMP = _load()


def _item(identifier: str, score: int, *, load_ok: bool = True, review: bool = False) -> dict:
    return {
        "id": identifier,
        "family": "given_then_ask",
        "load_ok": load_ok,
        "score": score,
        "pending_human_review": review,
    }


def _dimension(items: list[dict], name: str = "上下文与工作记忆") -> dict:
    scored = [row for row in items if not row["pending_human_review"] and row["load_ok"]]
    return {
        "name": name,
        "item_count": len(items),
        "error": None,
        "seconds": 1.0,
        "tally": {
            "machine_scored_items": len(scored),
            "machine_scored_correct": sum(int(row["score"]) for row in scored),
            "machine_normalised": None,
            "pending_human_review_items": len(items) - len(scored),
            "untested_items": 0,
        },
        "items": items,
    }


def _scores(prefix: str, values: list[int]) -> list[dict]:
    return [_item(f"{prefix}{index + 1:02d}", value) for index, value in enumerate(values)]


def _report(
    path: Path,
    *,
    dimensions: dict[str, dict],
    checkpoint_sha: str,
    eval_set_sha: str = "e" * 64,
    git_head: str = "h" * 40,
    chain: dict | None = None,
    trained: bool = False,
) -> Path:
    payload = {
        "format": "taiji-cap0-baseline-v1",
        "identity": {
            "git_head": git_head,
            "checkpoint_sha256": checkpoint_sha,
            "eval_set_sha256": eval_set_sha,
        },
        "checkpoint": "arm.pt",
        "declared_mode": "N",
        "trained_during_eval": trained,
        "chain": (
            chain
            if chain is not None
            else {"relax_legacy_guard": False, "constrained_decode": False}
        ),
        "scoring_discipline": list(DISCIPLINE),
        "dimensions": dimensions,
        "min_lines": {"C": 0.7},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def _arm_pair(
    tmp_path: Path,
    treated_dims: dict[str, dict],
    control_dims: dict[str, dict],
    *,
    control_eval_sha: str = "e" * 64,
    control_chain: dict | None = None,
    stem: str = "pair",
) -> tuple[Path, Path]:
    treated = _report(tmp_path / f"{stem}-t.json", dimensions=treated_dims, checkpoint_sha="a" * 64)
    control = _report(
        tmp_path / f"{stem}-c.json",
        dimensions=control_dims,
        checkpoint_sha="b" * 64,
        eval_set_sha=control_eval_sha,
        chain=control_chain,
    )
    return treated, control


def _run(tmp_path: Path, treated: Path, control: Path, name: str) -> tuple[int, dict]:
    out = tmp_path / name
    rc = CMP.main(
        [
            "--treated-report",
            str(treated),
            "--control-report",
            str(control),
            "--out-report",
            str(out),
        ]
    )
    return rc, json.loads(out.read_text(encoding="utf-8"))


#: 治疗臂对 C01–C04，对照臂只对 C05 ⇒ 正向翻转 4 枚、反向 1 枚、delta 恰为判据的 +3 那一格。
TREATED_C = [1, 1, 1, 1, 0, 0, 0, 0]
CONTROL_C = [0, 0, 0, 0, 1, 0, 0, 0]


def test_four_flips_against_one_is_a_gain_and_names_the_items(tmp_path: Path) -> None:
    treated, control = _arm_pair(
        tmp_path,
        {"C": _dimension(_scores("C", TREATED_C))},
        {"C": _dimension(_scores("C", CONTROL_C))},
        stem="gain",
    )
    rc, payload = _run(tmp_path, treated, control, "gain.json")
    assert rc == 0
    assert payload["format"] == "taiji-n5-22-cap0-pairing-v1"
    assert payload["verdict"] == "holds"
    assert payload["pairing_valid"] is True
    assert payload["direction"] == "capability_gain"
    assert payload["identity"]["treated_checkpoint_sha256"] == "a" * 64
    assert payload["identity"]["control_checkpoint_sha256"] == "b" * 64
    row = payload["dimensions"][0]
    assert row["enters_verdict"] is True
    assert row["machine_scored_items"] == {"treated": 8, "control": 8}
    assert row["flips_treated_only"] == 4
    assert row["flips_control_only"] == 1
    assert row["delta"] == 3
    assert row["flips_treated_only_ids"] == ["C01", "C02", "C03", "C04"]
    assert row["flips_control_only_ids"] == ["C05"]


def test_the_margin_is_strictly_three_not_two(tmp_path: Path) -> None:
    #: 阈值判据必须带"恰好等于"与"差一格"那一支，否则严格大于会被悄悄写成 ≥（J-N5u-3 的 ≥3）。
    for delta, expected in ((3, "capability_gain"), (2, "not_resolved")):
        control = [1 if index == 7 else 0 for index in range(8)]  # 反向一枚
        treated_scores = [1] * (delta + 1) + [0] * (7 - delta)
        treated, arm_control = _arm_pair(
            tmp_path,
            {"C": _dimension(_scores("C", treated_scores))},
            {"C": _dimension(_scores("C", control))},
            stem=f"margin{delta}",
        )
        rc, payload = _run(tmp_path, treated, arm_control, f"margin{delta}.json")
        row = payload["dimensions"][0]
        assert row["delta"] == delta
        assert rc == 0
        assert payload["direction"] == expected, (delta, payload["direction"])


def test_an_unequal_denominator_keeps_the_dimension_out(tmp_path: Path) -> None:
    #: 分母不等＝两把尺，减出来的数没意义（J-N5u-2）；配对资格仍成立，方向退成"无入场维"。
    treated, control = _arm_pair(
        tmp_path,
        {"C": _dimension(_scores("C", [1, 1, 1, 1, 1]))},
        {"C": _dimension(_scores("C", [0, 0, 0, 0, 0, 0]))},
        stem="denom",
    )
    rc, payload = _run(tmp_path, treated, control, "denom.json")
    assert rc == 0
    row = payload["dimensions"][0]
    assert row["machine_scored_items"] == {"treated": 5, "control": 6}
    assert row["denominators_equal"] is False
    assert row["enters_verdict"] is False
    assert payload["direction"] == "no_eligible_dimension"
    assert payload["failing_verdicts"] == ["at_least_one_eligible_dimension"]


def test_pending_and_unloaded_items_are_excluded_not_counted_as_wrong(tmp_path: Path) -> None:
    #: 待人工复核与装载失败的题两臂都不许进翻转；当成 0 分就是"把没测到写成测到且错"。
    treated_items = [_item("C01", 1), _item("C02", 0, review=True), _item("C03", 1, load_ok=False)]
    control_items = [_item("C01", 0), _item("C02", 1, review=True), _item("C03", 1, load_ok=False)]
    treated, control = _arm_pair(
        tmp_path,
        {"C": _dimension(treated_items)},
        {"C": _dimension(control_items)},
        stem="excl",
    )
    rc, payload = _run(tmp_path, treated, control, "excl.json")
    assert rc == 0
    row = payload["dimensions"][0]
    assert row["machine_scored_items"] == {"treated": 1, "control": 1}
    assert row["eligible_items"] == 1
    assert row["excluded_counts"]["pending_human_review"] == 1
    assert row["excluded_counts"]["load_failed"] == 1
    assert row["excluded_ids"]["pending_human_review"] == ["C02"]
    #: 只剩 C01 入场：delta 只有 +1 ⇒ 方向必须是 not_resolved，被排除的两题不许算成反向翻转。
    assert row["delta"] == 1
    assert payload["direction"] == "not_resolved"


def test_an_item_present_in_one_arm_only_is_published_not_guessed(tmp_path: Path) -> None:
    #: 两臂题集不同＝构成不等，缺失的 id 必须点名；把它当"对方 0 分"会凭空造出翻转。
    treated_items = _scores("C", [1, 1, 1, 1]) + [_item("C05", 1, review=True)]
    control_items = _scores("C", [0, 0, 0, 0])
    treated, control = _arm_pair(
        tmp_path,
        {"C": _dimension(treated_items)},
        {"C": _dimension(control_items)},
        stem="extra",
    )
    rc, payload = _run(tmp_path, treated, control, "extra.json")
    row = payload["dimensions"][0]
    assert rc == 0
    assert row["items_in_both_arms"] == 4
    assert row["excluded_ids"]["not_in_both_arms"] == ["C05"]
    assert row["flips_treated_only"] == 4
    assert payload["direction"] == "capability_gain"


def test_a_missing_eval_set_sha_kills_the_direction(tmp_path: Path) -> None:
    treated, control = _arm_pair(
        tmp_path,
        {"C": _dimension(_scores("C", TREATED_C))},
        {"C": _dimension(_scores("C", CONTROL_C))},
        control_eval_sha="f" * 64,
        stem="sha",
    )
    rc, payload = _run(tmp_path, treated, control, "sha.json")
    assert rc == 1
    assert payload["verdict"] == "rejected"
    assert payload["direction"] == "pairing_invalid"
    assert payload["failing_verdicts"] == ["eval_set_sha256_equal"]


def test_a_chain_flag_difference_is_a_pairing_failure(tmp_path: Path) -> None:
    #: 链路旗标（约束解码等）会换测面；两臂各用一条链时"能力差"根本无从比较。
    treated, control = _arm_pair(
        tmp_path,
        {"C": _dimension(_scores("C", TREATED_C))},
        {"C": _dimension(_scores("C", CONTROL_C))},
        control_chain={"relax_legacy_guard": False, "constrained_decode": True},
        stem="chain",
    )
    rc, payload = _run(tmp_path, treated, control, "chain.json")
    assert rc == 1
    assert payload["failing_verdicts"] == ["chain_equal"]


def test_opposite_deltas_across_dimensions_are_inconsistent_not_selectable(tmp_path: Path) -> None:
    #: 一维 +3、一维 −4 ⇒ 不许挑一支写成增益（分支互斥在这里真的挡住事）。
    treated, control = _arm_pair(
        tmp_path,
        {
            "C": _dimension(_scores("C", TREATED_C)),
            "D": _dimension(_scores("D", [0, 0, 0, 0, 0, 0, 0, 0]), name="简单指令与推理"),
        },
        {
            "C": _dimension(_scores("C", CONTROL_C)),
            "D": _dimension(_scores("D", [1, 1, 1, 1, 0, 0, 0, 0]), name="简单指令与推理"),
        },
        stem="mixed",
    )
    rc, payload = _run(tmp_path, treated, control, "mixed.json")
    assert rc == 0
    deltas = {row["dimension"]: row["delta"] for row in payload["dimensions"]}
    assert deltas == {"C": 3, "D": -4}
    assert payload["direction"] == "direction_inconsistent"


def test_dimensions_without_a_tally_are_not_compared_but_are_listed(tmp_path: Path) -> None:
    #: A／F／H 那类 `status=not_executed` 的维度没有 tally ⇒ 不参与配对，但要出版"没比"的名单。
    not_executed = {"name": "模型真实性", "status": "not_executed", "item_count": 6}
    treated, control = _arm_pair(
        tmp_path,
        {"C": _dimension(_scores("C", TREATED_C)), "A": not_executed},
        {"C": _dimension(_scores("C", CONTROL_C)), "A": not_executed},
        stem="notexec",
    )
    rc, payload = _run(tmp_path, treated, control, "notexec.json")
    assert rc == 0
    assert [row["dimension"] for row in payload["dimensions"]] == ["C"]
    assert payload["not_compared_dimensions"] == ["A"]


def test_missing_or_foreign_json_refuses_and_existing_file_is_protected(tmp_path: Path) -> None:
    treated, control = _arm_pair(
        tmp_path,
        {"C": _dimension(_scores("C", TREATED_C))},
        {"C": _dimension(_scores("C", CONTROL_C))},
        stem="guard",
    )
    rc = CMP.main(
        [
            "--treated-report",
            str(treated),
            "--control-report",
            str(tmp_path / "never-was.json"),
            "--out-report",
            str(tmp_path / "x.json"),
        ]
    )
    assert rc == 2
    foreign = tmp_path / "foreign.json"
    foreign.write_text('{"format": "other"}', encoding="utf-8", newline="\n")
    rc_foreign = CMP.main(
        [
            "--treated-report",
            str(treated),
            "--control-report",
            str(foreign),
            "--out-report",
            str(tmp_path / "y.json"),
        ]
    )
    assert rc_foreign == 2
    out = tmp_path / "first.json"
    _run(tmp_path, treated, control, "first.json")
    rc_overwrite = CMP.main(
        [
            "--treated-report",
            str(treated),
            "--control-report",
            str(control),
            "--out-report",
            str(out),
        ]
    )
    assert rc_overwrite == 2
    assert json.loads(out.read_text(encoding="utf-8"))["direction"] == "capability_gain"


def test_help_walks(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        CMP.main(["--help"])
    assert exit_info.value.code == 0
