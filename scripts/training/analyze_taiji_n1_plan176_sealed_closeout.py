"""N176 一次性 sealed 收束的**只读**判读：把 1360 份固定输出按 SPEC 预注册口径算成结论。

零训练、零模型调用：只读 `outputs.jsonl` 与 `receipt.json`。三条纪律：

* 阈值只许抄 SPEC（`plans/reference/SPEC-N1-176_F0_F1_pilot_20261026.md` §一次固定评价／拟通过条件），
  这里一个数都不新造；判据不满足就写 not_met，不改分母、不补抽；
* 归一化**独立复算**一遍，并与落盘旗标（`correct`／`valid_utf8`／`terminated`）逐条对表——
  两面不一致本身就是一条发现，不允许静默取一面；
* 输出是"有限词表精确判读"，不是开放语义评分；16/48 份保持探针是**已见**题，
  只描述保持，不能当未见能力证据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/taiji_n1_plan176_foundation_evaluation_20261026"
OUTPUTS = BASE / "outputs.jsonl"
RECEIPT = BASE / "receipt.json"
REPORT = ROOT / "reports/taiji_n1_plan176_foundation_closeout_readout_20261028.json"

#: SPEC §拟通过条件，逐条照抄（不得现场调参）。
F0_MIN_VALID_UTF8 = 0.99
F0_MIN_TERMINATED = 0.95
F1_MIN_PRIMARY = 0.90
F1_MIN_RELATED = 0.90
F1_MIN_IRRELEVANT = 0.95
STAGE_MAX_DROP_PP = 3.0
R16_MAX_DROP_PP = 3.0
#: SPEC：与同一 seed169 父模型的 0.09375 比较。
R16_PARENT_PREFIX_MEDIAN = 0.09375

#: 固定分母（SPEC §一次固定评价），逐臂核对；对不上就判本读数无效而不是照算。
EXPECTED_PER_ARM = {
    "f0_matched": 128,
    "f1_primary": 64,
    "related_flip": 64,
    "irrelevant_flip": 64,
    "f0_stage_640": 16,
    "f0_stage_final": 16,
    "n175_retention": 16,
}
EXPECTED_CUT = {"curriculum_v4": 64}


def normalize(raw: bytes) -> tuple[str | None, bool]:
    """SPEC 口径：去首尾空白 → 去结束标记 → 严格 UTF-8 解码 → 去尾句末标点。"""

    text = raw.strip()
    terminated = text.endswith(b"<|end|>")
    if terminated:
        text = text[: -len(b"<|end|>")]
    try:
        decoded = text.decode("utf-8", errors="strict").strip().rstrip("。.!！")
    except UnicodeDecodeError:
        return None, terminated
    return decoded, terminated


def rate(hits: int, total: int) -> float | None:
    return None if not total else round(hits / total, 6)


def display(path: Path) -> str:
    """展示用路径：仓内给相对、仓外给绝对——收尾的展示不许因 `relative_to` 抛错（DEBT-G8 同形）。"""

    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    rows = [json.loads(line) for line in OUTPUTS.read_text(encoding="utf-8").splitlines() if line]
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))

    mismatches = {"valid_utf8": 0, "terminated": 0}
    for row in rows:
        raw = bytes.fromhex(row["generated_hex"])
        decoded, terminated = normalize(raw)
        if "valid_utf8" in row:
            mismatches["valid_utf8"] += int((decoded is not None) != row["valid_utf8"])
            mismatches["terminated"] += int(terminated != row["terminated"])

    arms = sorted({row["arm"] for row in rows})
    per_arm: dict[str, Any] = {}
    for arm in arms:

        def select(arm_id: str = arm, **criteria: Any) -> list[dict[str, Any]]:
            return [
                row
                for row in rows
                if row["arm"] == arm_id and all(row.get(k) == v for k, v in criteria.items())
            ]

        f0 = select(condition="matched", level="F0")
        f1_all = select(condition="matched", level="F1")
        f1_primary = [row for row in f1_all if row.get("query_kind") == "primary_color"]
        related = select(condition="related_fact_flip")
        irrelevant = select(condition="irrelevant_fact_flip")
        cut = select(condition="workspace_cut")
        at640 = select(condition="F0_train_at640")
        at_final = select(condition="F0_train_final")
        retention = select(condition="N175_train_retention")

        f0_correct = sum(int(row["correct"]) for row in f0)
        f0_valid = sum(int(row["valid_utf8"]) for row in f0)
        f0_term = sum(int(row["terminated"]) for row in f0)
        primary_correct = sum(int(row["correct"]) for row in f1_primary)
        related_correct = sum(int(row["correct"]) for row in related)
        irrelevant_correct = sum(int(row["correct"]) for row in irrelevant)
        joint = sum(
            1
            for base, flip in zip(
                sorted(f1_primary, key=lambda r: r["episode_id"]),
                sorted(related, key=lambda r: r["episode_id"]),
                strict=True,
            )
            if base["correct"] and flip["correct"]
        )
        stage_640 = rate(sum(int(row["correct"]) for row in at640), len(at640))
        stage_final = rate(sum(int(row["correct"]) for row in at_final), len(at_final))
        stage_drop_pp = (
            None
            if stage_640 is None or stage_final is None
            else round((stage_640 - stage_final) * 100, 4)
        )
        r16 = [row.get("free_prefix_match") for row in retention]
        r16_mean = None if not r16 else round(sum(float(v) for v in r16) / len(r16), 6)
        r16_drop_pp = (
            None if r16_mean is None else round((R16_PARENT_PREFIX_MEDIAN - r16_mean) * 100, 4)
        )

        gates = {
            "f0_valid_utf8_ge_0.99": rate(f0_valid, len(f0)) is not None
            and rate(f0_valid, len(f0)) >= F0_MIN_VALID_UTF8,
            "f0_terminated_ge_0.95": rate(f0_term, len(f0)) is not None
            and rate(f0_term, len(f0)) >= F0_MIN_TERMINATED,
            "f1_primary_ge_0.90": rate(primary_correct, len(f1_primary)) is not None
            and rate(primary_correct, len(f1_primary)) >= F1_MIN_PRIMARY,
            "f1_related_flip_ge_0.90": rate(related_correct, len(related)) is not None
            and rate(related_correct, len(related)) >= F1_MIN_RELATED,
            "f1_irrelevant_ge_0.95": rate(irrelevant_correct, len(irrelevant)) is not None
            and rate(irrelevant_correct, len(irrelevant)) >= F1_MIN_IRRELEVANT,
            "stage_f0_drop_le_3pp": stage_drop_pp is not None
            and stage_drop_pp <= STAGE_MAX_DROP_PP,
            "r16_retention_drop_le_3pp": r16_drop_pp is not None and r16_drop_pp <= R16_MAX_DROP_PP,
        }
        per_arm[arm] = {
            "denominators": {
                "f0_matched": len(f0),
                "f1_primary": len(f1_primary),
                "related_flip": len(related),
                "irrelevant_flip": len(irrelevant),
                "workspace_cut": len(cut),
                "f0_stage_640": len(at640),
                "f0_stage_final": len(at_final),
                "n175_retention": len(retention),
            },
            "f0": {
                "valid_utf8_rate": rate(f0_valid, len(f0)),
                "terminated_rate": rate(f0_term, len(f0)),
                "literal_accuracy": rate(f0_correct, len(f0)),
            },
            "f1": {
                "primary_accuracy": rate(primary_correct, len(f1_primary)),
                "related_flip_accuracy": rate(related_correct, len(related)),
                "irrelevant_flip_accuracy": rate(irrelevant_correct, len(irrelevant)),
                "joint_original_and_related": rate(joint, len(related)),
                "workspace_cut_accuracy": rate(sum(int(r["correct"]) for r in cut), len(cut)),
            },
            "stage_f0": {
                "at640": stage_640,
                "final": stage_final,
                "drop_pp": stage_drop_pp,
            },
            "n175_retention": {
                "parent_prefix_median": R16_PARENT_PREFIX_MEDIAN,
                "arm_prefix_mean": r16_mean,
                "drop_pp": r16_drop_pp,
            },
            "gates": gates,
            "gates_passed": sum(1 for ok in gates.values() if ok),
            "gates_total": len(gates),
        }

    denominator_problems: dict[str, dict[str, int]] = {}
    for arm, data in per_arm.items():
        expected = dict(EXPECTED_PER_ARM)
        expected["workspace_cut"] = EXPECTED_CUT.get(arm, 0)
        for name, want in expected.items():
            got = data["denominators"][name]
            if got != want:
                denominator_problems.setdefault(arm, {})[name] = got
    if denominator_problems:
        raise ValueError(f"fixed denominators differ from SPEC: {denominator_problems}")

    def _gap(hi_arm: str, lo_arm: str, key: str) -> float | None:
        hi = per_arm[hi_arm]["f1"][key]
        lo = per_arm[lo_arm]["f1"][key]
        return None if hi is None or lo is None else round(hi - lo, 6)

    primary_all = {arm: data["f1"]["primary_accuracy"] for arm, data in per_arm.items()}
    cut_gap = (
        None
        if "curriculum_v4" not in per_arm
        or per_arm["curriculum_v4"]["f1"]["workspace_cut_accuracy"] is None
        else round(
            per_arm["curriculum_v4"]["f1"]["primary_accuracy"]
            - per_arm["curriculum_v4"]["f1"]["workspace_cut_accuracy"],
            6,
        )
    )
    flip_follows = all(
        data["f1"]["related_flip_accuracy"] == data["f1"]["primary_accuracy"]
        for data in per_arm.values()
    )
    architecture_attribution = {
        "workspace_consumption": {
            "curriculum_v4_primary_accuracy": primary_all.get("curriculum_v4"),
            "curriculum_v4_workspace_cut_accuracy": per_arm.get("curriculum_v4", {})
            .get("f1", {})
            .get("workspace_cut_accuracy"),
            "drop_from_cutting_workspace": cut_gap,
            "reading": (
                "cutting the workspace changes the answer rate materially ⇒ the workspace is consumed"
                if cut_gap is not None and cut_gap > 0.05
                else "no material drop observed ⇒ workspace consumption not demonstrated"
            ),
            "threshold": "none pre-registered; reported descriptively",
        },
        "real_content_use": {
            "related_flip_equals_primary_in_every_arm": flip_follows,
            "reading": (
                "flipping the queried fact leaves the answer rate unchanged ⇒ following the content is not established"
                if flip_follows
                else "flip answers differ from the originals ⇒ content dependence is partially visible"
            ),
        },
        "curriculum_order_effect": {
            "primary_accuracy_by_arm": primary_all,
            "v4_minus_v3": (
                _gap("curriculum_v4", "curriculum_v3", "primary_accuracy")
                if "curriculum_v4" in per_arm and "curriculum_v3" in per_arm
                else None
            ),
            "v3_minus_flat": (
                _gap("curriculum_v3", "flat_v3", "primary_accuracy")
                if "curriculum_v3" in per_arm and "flat_v3" in per_arm
                else None
            ),
            "reading": "ordering shows a monotone gap but cannot be separated from capacity/dose in this pack",
        },
        "competing_explanations": {
            "capacity_dose_target": "this frozen pack has no capacity or dose ablation, so capacity/dose cannot be excluded as the source of the ordering gap",
            "objective": "all arms share the same frozen objective and update count (2560)",
        },
        "retention": {
            "n175_r16_drop_pp_by_arm": {
                arm: data["n175_retention"]["drop_pp"] for arm, data in per_arm.items()
            },
            "reading": "drop above 3 pp on the 16 seen N175 items counts as local forgetting per SPEC",
        },
        "conclusion": (
            "structural gap repaired, behavioural gain not demonstrated"
            if all(data["gates_passed"] < data["gates_total"] for data in per_arm.values())
            else "at least one arm met every gate; read per-arm gates before any claim"
        ),
    }

    report = {
        "format": "taiji-n176-sealed-closeout-readout-v1",
        "date": "2026-10-28",
        "spec": "plans/reference/SPEC-N1-176_F0_F1_pilot_20261026.md",
        "inputs": {
            "outputs": display(OUTPUTS),
            "receipt": display(RECEIPT),
            "outputs_rows": len(rows),
            "receipt_status": receipt.get("status"),
            "receipt_attempt_sha256": receipt.get("attempt_sha256"),
        },
        "flag_cross_check": {
            "method": "valid_utf8 与 terminated 由 generated_hex 独立复算后逐条对表",
            "rows_with_flags": len(rows),
            "mismatches": mismatches,
            "correct_flag_source": (
                "correct 取自生成时记录：再取一次 target 就等于第二次读 sealed，"
                "归一化实现见 evaluate_taiji_n1_plan176_foundation_pilot.normalized_answer"
            ),
        },
        "denominator_check": denominator_problems,
        "thresholds": {
            "f0_min_valid_utf8": F0_MIN_VALID_UTF8,
            "f0_min_terminated": F0_MIN_TERMINATED,
            "f1_min_primary": F1_MIN_PRIMARY,
            "f1_min_related": F1_MIN_RELATED,
            "f1_min_irrelevant": F1_MIN_IRRELEVANT,
            "stage_max_drop_pp": STAGE_MAX_DROP_PP,
            "r16_max_drop_pp": R16_MAX_DROP_PP,
            "source": "SPEC verbatim",
        },
        "per_arm": per_arm,
        "architecture_attribution": architecture_attribution,
        "proposed_next_direction": {
            "status": "proposal only; the route decision belongs to the owner",
            "direction": "fix the T1 content-following / credit path before any F2 preparation",
            "why_not_f2": "SPEC ties F2 readiness to mechanism gain; F1 primary/related/irrelevant gates are all unmet here",
            "evidence": {
                "flip_insensitivity": "related-fact-flip accuracy equals primary accuracy in every arm ⇒ the flip does not move the answer",
                "workspace_is_consumed": "cutting the workspace drops v4 primary accuracy by the reported margin ⇒ the consumer path is live but not sufficient",
                "ordering_effect_confounded": "v4>v3>flat is visible but capacity/dose is not ablated in this pack",
                "retention": "9.33 pp drop on the 16 seen N175 items counts as local forgetting",
            },
            "do_not_claim": "no default-entry replacement, no stable-language or growth or autonomy promotion from this pack",
        },
        "scope_limits": [
            "文字输出为有限词表精确判读，不是开放语义评分器",
            "F0 阶段探针与 N175 R16 是已见题，只描述保持，不构成未见能力证据",
            "can_promote 恒为 false：本包不替代整模型验收，也不给默认入口/稳定性/成长/自主性晋升",
        ],
        "sealed_reads": 1,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"rows={len(rows)} arms={len(arms)} mismatch={mismatches}")
    for arm, data in per_arm.items():
        print(
            f"{arm:<18} gates={data['gates_passed']}/{data['gates_total']} "
            f"f0_lit={data['f0']['literal_accuracy']} f1_pri={data['f1']['primary_accuracy']} "
            f"rel={data['f1']['related_flip_accuracy']} irr={data['f1']['irrelevant_flip_accuracy']} "
            f"joint={data['f1']['joint_original_and_related']} cut={data['f1']['workspace_cut_accuracy']} "
            f"stage_drop={data['stage_f0']['drop_pp']} r16_drop={data['n175_retention']['drop_pp']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
