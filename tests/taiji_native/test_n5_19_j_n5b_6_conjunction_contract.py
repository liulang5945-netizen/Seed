"""PLAN-N5-09 的四元合取契约测（`adjudicate_taiji_n5_shadow_gate.py`）。

三态归类逐支点名，混合态（G-N5h-3）单独测——那是本件最容易写错的一格：
一支未测＋一支已判否证时，合取只能说"未判"，不能说"不成立"。
兼容锚按㊵-663 起草时自纠后的形状走（"不接线时逐字节相同"是一条永远不为假的守卫，已废弃）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n5_shadow_gate.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_conjunction", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = _load()


def _four(**over: Any) -> dict[str, Any]:
    values = {
        "j_n5b_2": "shadow_learned",
        "j_n5b_3": "valid",
        "j_n5b_4": "holds",
        "j_n5b_5": "holds",
    }
    values.update(over)
    return values


def test_all_four_established_is_the_only_way_to_leave_the_conjunction_open() -> None:
    out = JUDGE.judge_conjunction(_four())
    assert out["j_n5b_6"] == "established"
    assert out["missing"] == [] and out["unmeasured"] == []


def test_ruler_unusable_counts_as_not_established_not_as_unmeasured() -> None:
    #: PLAN-N5-02 §4 :52 原文把这条限制在"结论"上：尺答不了 ⇒ 这一支不成立，而不是"还没测"。
    out = JUDGE.judge_conjunction(_four(j_n5b_4="ruler_unusable"))
    assert out["j_n5b_6"] == "not_established"
    assert out["missing"] == ["J-N5b-4"]
    assert out["unmeasured"] == []
    assert out["branches"]["j_n5b_4"] == {
        "criterion": "J-N5b-4",
        "value": "ruler_unusable",
        "state": "not_established",
    }


def test_mixed_state_prefers_unverified_over_not_established() -> None:
    #: G-N5h-3 的反例支：一支未测＋一支已判否证 ⇒ 只能说"未判"，`missing` 必须留空。
    #: 若这里写成 not_established，就等于把"还没测"混进"测了且否证"——DEBT-G89 那句假陈述的成因。
    out = JUDGE.judge_conjunction(_four(j_n5b_4="ruler_unusable", j_n5b_5="unverified_noise_floor_missing"))
    assert out["j_n5b_6"] == "unverified"
    assert out["missing"] == []
    assert out["unmeasured"] == ["J-N5b-5"]


def test_today_two_arm_values_are_expected_not_established_naming_two_branches() -> None:
    #: §6 落笔即钉的"预期为绿成员"：今日真值（2 learned／3 valid／4 ruler_unusable／5 cost_persists）
    #: 应判 `not_established`，点名 J-N5b-4 与 J-N5b-5 两支。
    out = JUDGE.judge_conjunction(
        _four(j_n5b_4="ruler_unusable", j_n5b_5="cost_persists")
    )
    assert out["j_n5b_6"] == "not_established"
    assert out["missing"] == ["J-N5b-4", "J-N5b-5"]


def test_arm_dependent_is_a_judgement_not_a_missing_measurement() -> None:
    out = JUDGE.judge_conjunction(_four(j_n5b_5="arm_dependent"))
    assert out["j_n5b_6"] == "not_established"
    assert out["missing"] == ["J-N5b-5"]


def test_the_two_not_judgable_retention_forms_are_unmeasured(tmp_path: Path) -> None:
    #: `not_judgable_below_min`／`not_resolved_insufficient_range` 是"这次判不了"，不是阴性结果。
    for value in ("not_judgable_below_min", "not_resolved_insufficient_range"):
        out = JUDGE.judge_conjunction(_four(j_n5b_5=value))
        assert out["j_n5b_6"] == "unverified", value
        assert out["unmeasured"] == ["J-N5b-5"], value


def test_an_unlisted_value_never_gets_a_silent_classification() -> None:
    #: G-N5h-5：白名单外的值既不升成立也不判否证，而是点名＋让 rc 抬起。
    out = JUDGE.judge_conjunction(_four(j_n5b_2="shadow_learned_but_maybe"))
    assert out["j_n5b_6"] == "unverified"
    assert out["unknown_values"] == ["J-N5b-2='shadow_learned_but_maybe'"]
    assert out["unmeasured"] == ["J-N5b-2"]


def test_unverified_prefix_family_is_classified_without_being_enumerated() -> None:
    out = JUDGE.judge_conjunction(_four(j_n5b_2="unverified_missing_face"))
    assert out["j_n5b_6"] == "unverified"
    assert out["unknown_values"] == []
    assert out["unmeasured"] == ["J-N5b-2"]


def test_branch_original_values_are_published_untranslated() -> None:
    #: §6 发表资格前置①：原值不翻译地出版，否则"三态"会盖掉读数本身。
    out = JUDGE.judge_conjunction(_four(j_n5b_3="invalid"))
    assert out["branches"]["j_n5b_3"]["value"] == "invalid"
    assert out["j_n5b_6"] == "not_established"


def test_missing_branch_key_is_unmeasured_not_assumed_zero() -> None:
    #: 少给一枚键＝仪器没算到它，不许 `.get` 猜成任何已列举值（㊵-590 的 `get()` 假 null 同族）。
    values = _four()
    del values["j_n5b_3"]
    out = JUDGE.judge_conjunction(values)
    assert out["j_n5b_6"] == "unverified"
    assert out["unmeasured"] == ["J-N5b-3"]
    assert out["branches"]["j_n5b_3"]["value"] is None


@pytest.mark.parametrize("value", ["holds", "not_holds", "ruler_unusable"])
def test_every_listed_j_n5b_4_value_has_exactly_one_state(value: str) -> None:
    out = JUDGE.judge_conjunction(_four(j_n5b_4=value))
    state = out["branches"]["j_n5b_4"]["state"]
    assert state in {"established", "not_established", "unmeasured"}
    assert (state == "established") is (value == "holds")


def test_the_sealed_pre_wiring_artifact_is_not_regenerated(tmp_path: Path) -> None:
    """G-N5h-4(b)：接线不许回头改写已入库的读数件。

    ㊵-664 起草时我先把这条锚写成"不接线时逐字节相同"——合取一旦接上就没有"不接线"那条分支，
    那是一条**永远不为假**的守卫（[[guard-must-be-able-to-fail]]）。现行形状＝钉住接线前那份件的摘要。
    """
    sealed = REPO / "reports" / "taiji_n5_08_jn5b5_with_noise_floor_20261010.json"
    digest = hashlib.sha256(sealed.read_bytes()).hexdigest()
    assert digest == "0298ed15beadc077a2e13abc147d043a6eba5f2c841764441eb03c70b9f79523"
    payload = json.loads(sealed.read_text(encoding="utf-8"))
    #: (c) 只加键不删键：接线前后顶层键名的差集只允许是新增。
    assert {"arms", "format", "j_n5b_4", "j_n5b_5", "j_n5b_6", "pairing", "rc"} <= set(payload)
    #: (a) 四枚成员键的**原值**不受接线影响：这份件里 6 还是旧字面量，成员值与我今日真读一致。
    assert payload["j_n5b_4"] == "ruler_unusable"
    assert payload["j_n5b_5"] == "cost_persists"
    assert payload["j_n5b_6"] == "not_adjudicable_until_2_3_4_5_are_all_measured"
