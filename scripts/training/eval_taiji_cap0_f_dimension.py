"""CAP F 维（项目代表能力）的证据复判与统一入口现场重跑。

07 §4.2 对 F 的要求是"至少一项能力满足其独立冻结门，**并展示输入到实际结果**，
不仅报告局部 probe"。本模块把它拆成两半：

- `adjudicate_f_items`：把评价集 v1 的 F 四项各自的门文本读回**被引报告里的原始数字**重算，
  而不是只看那份报告在不在盘上。文件在场不等于门过了。
- `run_unified_entry_live`：现场重跑统一入口证据包（零训练、约 11 秒），产出"输入→实际动作
  →目标达成"的逐臂读数，并对冻结线 L1–L4 复算。

两条纪律贯穿本模块：**拆不出可机检形态的门文本记 `unverified`，不等于 `pass`**；封存报告
一律不覆写（预注册 §4），现场重跑的产物只落临时目录。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 统一入口证据包 runner 与其冻结预注册（现场重跑复用同一份实现，不另写一套判据）。
UNIFIED_ENTRY_RUNNER = PROJECT_ROOT / "scripts" / "training" / "run_taiji_unified_entry_evidence.py"
UNIFIED_ENTRY_SEALED_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_unified_entry_evidence_20260919.json"
)
#: 预注册 §2 的资源线（每臂 ≤120s、总 sweep ≤600s）；超出即判执行失败，不等结果调线。
UNIFIED_ENTRY_WALL_CAP = 600.0


def _clause(label: str, held: bool | None, detail: str) -> dict[str, Any]:
    """一条门文本子句的复算结果。`held=None` 表示这条读不到可机检形态 ⇒ unverified。"""

    return {"clause": label, "held": held, "detail": detail}


def _verdict_of(clauses: list[dict[str, Any]]) -> str:
    if any(c["held"] is False for c in clauses):
        return "fail"
    if any(c["held"] is None for c in clauses):
        return "partial"
    return "pass" if clauses else "partial"


def _gates_all_true(report: dict[str, Any], prefix: str) -> tuple[dict[str, bool], int]:
    gates = {
        str(key): bool(value)
        for key, value in (report.get("gates") or {}).items()
        if str(key).startswith(prefix)
    }
    return gates, len(gates)


def _adjudicate_f01(report: dict[str, Any]) -> list[dict[str, Any]]:
    """门文本：六门全过 + outcome=representation_discriminative。"""

    gates, count = _gates_all_true(report, "")
    clauses = [
        _clause(
            "六门全过",
            (count == 6 and all(gates.values())) if count else None,
            f"gates={gates}",
        ),
        _clause(
            "outcome=representation_discriminative",
            bool(report.get("outcome")) and report["outcome"] == "representation_discriminative",
            f"outcome={report.get('outcome')!r}",
        ),
    ]
    if count and count != 6:
        clauses.append(
            _clause("报告门数与冻结文本相符", False, f"报告只有 {count} 门，冻结文本要求 6 门")
        )
    return clauses


def _adjudicate_f02(report: dict[str, Any]) -> list[dict[str, Any]]:
    """门文本：G1-G6 全过。当前实测 G4/G5 不过 ⇒ 该项**必须**判 fail，负结果不改绿。"""

    gates, count = _gates_all_true(report, "G")
    failed = sorted(key for key, value in gates.items() if not value)
    return [
        _clause(
            "G1-G6 全过",
            (count == 6 and not failed) if count else None,
            f"gates={gates}" if not failed else f"未过：{failed}",
        ),
        _clause(
            "experiment_passed 与门一致",
            bool(report.get("gates")) and bool(report.get("experiment_passed")) is not None,
            f"experiment_passed={report.get('experiment_passed')!r}",
        ),
    ]


def _adjudicate_f03(report: dict[str, Any]) -> list[dict[str, Any]]:
    """门文本：create 行三格 +2.000、interleaved 6/6、零回归。

    `+2.000` 与 `interleaved 6/6` 在报告里都没有可定位的对应字段 ⇒ 记 unverified 而不是
    猜一个字段当作它。可机检的只有 verdict 块的三格归属与回归清单。
    """

    verdict = report.get("verdict") or {}
    positive = sorted(str(cell) for cell in verdict.get("positive_gain_cells") or [])
    create_cells = [cell for cell in positive if cell.startswith("create__")]
    clauses = [
        _clause(
            "create 行三格为正增益",
            (len(create_cells) == 3 and len(positive) == 3) if positive else None,
            f"positive_gain_cells={positive}",
        ),
        _clause(
            "零回归（m4_regresses_cells 为空）",
            isinstance(verdict.get("m4_regresses_cells"), list)
            and not verdict["m4_regresses_cells"],
            f"m4_regresses_cells={verdict.get('m4_regresses_cells')!r}",
        ),
        _clause(
            "无未解释变化格（cells_with_unexplained_change 为空）",
            isinstance(verdict.get("cells_with_unexplained_change"), list)
            and not verdict["cells_with_unexplained_change"],
            f"cells_with_unexplained_change={verdict.get('cells_with_unexplained_change')!r}",
        ),
        _clause(
            "create 行三格增益值 = +2.000",
            None,
            "报告 verdict 块只给格名与计数，未给逐格增益数值 ⇒ 该子句无法从封存件机检",
        ),
        _clause(
            "interleaved 6/6",
            None,
            "报告内 interleaved 计数只出现在 frozen/audited=0 的审计块，与门文本所指的 6/6 "
            "不是同一量 ⇒ 该子句无法从封存件机检",
        ),
    ]
    return clauses


def _adjudicate_f04(report: dict[str, Any]) -> list[dict[str, Any]]:
    """门文本：默认入口可加载并产出原始输出。must_show 要求同时披露默认 tick 与链路缺陷。"""

    reality = report.get("model_reality") or {}
    default_entry = (report.get("raw_output_inventory") or {}).get("default_entry") or {}
    bytes_out = default_entry.get("total_output_bytes")
    clauses = [
        _clause(
            "默认入口可加载",
            bool(reality.get("load_ok")) is True,
            f"load_ok={reality.get('load_ok')!r} load_error={reality.get('load_error')!r}",
        ),
        _clause(
            "产出原始输出",
            isinstance(bytes_out, int) and bytes_out > 0,
            f"total_output_bytes={bytes_out!r} turns_answered={default_entry.get('turns_answered')!r}",
        ),
    ]
    template = default_entry.get("template_signature") or {}
    clauses.append(
        _clause(
            "输出非固定模板回显（must_show：不得把模板归因模型）",
            template.get("templated") is False if template else None,
            f"templated={template.get('templated')!r} distinct_signatures="
            f"{template.get('distinct_signatures')!r}",
        )
    )
    clauses.append(
        _clause(
            "默认入口服务的确实是训练态（must_show 披露项）",
            bool(reality.get("wiring_defect")) is False,
            f"default_tick={reality.get('default_tick')!r} "
            f"most_trained_tick={reality.get('most_trained_tick')!r} "
            f"wiring_defect={reality.get('wiring_defect')!r}",
        )
    )
    return clauses


F_ADJUDICATORS = {
    "F01": _adjudicate_f01,
    "F02": _adjudicate_f02,
    "F03": _adjudicate_f03,
    "F04": _adjudicate_f04,
}


def _must_show_f01(report: dict[str, Any]) -> list[dict[str, Any]]:
    """must_show：输入到实际结果的完整链，不得只报局部 probe。

    可机检的形式是：候选格（不参与拟合的那三格）有**实际执行**出来的逐对增益，而不只是拟合侧
    的系数/预测。拟合侧单独在场 ⇒ 该子句不成立。
    """

    corpus = report.get("corpus") or {}
    surface = (report.get("discrimination") or {}).get("candidate_surface") or {}
    actual = surface.get("actual_gain_vs_all_singleton_oracle") or {}
    return [
        _clause(
            "候选格有实际执行的 episode（非仅拟合）",
            int((corpus.get("episodes") or {}).get("candidate") or 0) > 0,
            f"episodes.candidate={(corpus.get('episodes') or {}).get('candidate')!r}",
        ),
        _clause(
            "逐对**实际**增益在册（输入→结果，不是预测表）",
            len(actual) > 0,
            f"actual_gain 条目数={len(actual)}",
        ),
    ]


def adjudicate_f_items(
    f_items: list[dict[str, Any]], root: Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    """逐 F 项：读回被引报告的原始数字重算门，替换"报告在不在盘上"这一空洞判据。"""

    rows: list[dict[str, Any]] = []
    for item in f_items:
        item_id = str(item.get("id"))
        reference = str(item.get("reference", ""))
        path = root / reference
        row: dict[str, Any] = {
            "id": item_id,
            "capability": item.get("capability"),
            "gate": item.get("gate"),
            "must_show": item.get("must_show"),
            "report": reference,
            "report_present": path.is_file(),
        }
        adjudicator = F_ADJUDICATORS.get(item_id)
        if not row["report_present"]:
            row["clauses"] = [_clause("被引报告在场", False, f"{reference} 不存在")]
            row["verdict"] = "unavailable"
        elif adjudicator is None:
            row["clauses"] = [
                _clause("该项有复算器", False, f"没有 {item_id} 的复算实现 ⇒ 不得记通过")
            ]
            row["verdict"] = "not_adjudicated"
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                row["clauses"] = [_clause("被引报告可读", False, f"{type(exc).__name__}: {exc}")]
                row["verdict"] = "unavailable"
            else:
                clauses = adjudicator(payload)
                row["clauses"] = clauses
                row["gate_verdict"] = _verdict_of(clauses)
                must_show = _MUST_SHOW_ADJUDICATORS.get(item_id)
                if must_show is None:
                    row["must_show_clauses"] = [
                        _clause(
                            "must_show 链已机检",
                            None,
                            "该项的 must_show 没有机检实现 ⇒ 至多 partial，不得记为已展示输入到实际结果",
                        )
                    ]
                else:
                    row["must_show_clauses"] = must_show(payload)
                row["verdict"] = (
                    "pass"
                    if row["gate_verdict"] == "pass"
                    and _verdict_of(row["must_show_clauses"]) == "pass"
                    else row["gate_verdict"] if row["gate_verdict"] == "fail" else "partial"
                )
                row["report_recorded_outcome"] = payload.get("outcome")
        rows.append(row)
    return rows


_MUST_SHOW_ADJUDICATORS = {
    "F01": _must_show_f01,
}
#: F02/F03/F04 的 must_show 尚未机检：门已过到那一步时它们只会停在 partial。


def adjudicate_unified_entry(report: dict[str, Any]) -> dict[str, Any]:
    """按预注册 §2（含 §5/§6 两次修正）从逐臂原始数字复算 L1–L4，不信任报告自带结论。"""

    per_arm = {str(row["arm"]): row for row in report.get("per_arm") or []}
    full = per_arm.get("full") or {}
    others = {name: row for name, row in per_arm.items() if name != "full"}
    disable_selection = per_arm.get("disable_selection") or {}

    l1_arm_rates = {name: row.get("main_success_rate") for name, row in others.items()}
    l1 = all(
        isinstance(full.get("main_success_rate"), (int, float))
        and isinstance(rate, (int, float))
        and full["main_success_rate"] >= rate
        for rate in l1_arm_rates.values()
    ) and bool(others)

    l2 = (
        isinstance(full.get("total_steps_with_injection"), int)
        and isinstance(disable_selection.get("total_steps_with_injection"), int)
        and full["total_steps_with_injection"] <= disable_selection["total_steps_with_injection"]
    )

    l3 = (
        isinstance(full.get("main_success_rate"), (int, float))
        and full["main_success_rate"] >= 2 / 3
    )

    l4 = bool(per_arm) and all(row.get("all_trace_valid") is True for row in per_arm.values())

    recomputed = {
        "L1_full_beats_all_ablations": l1,
        "L2_handoff_resource_line": l2,
        "L3_failure_injection_still_goals": l3,
        "L4_trace_and_safety": l4,
    }
    recorded = {str(k): bool(v) for k, v in (report.get("lines") or {}).items()}
    unverifiable = [
        "L3:预注册文本要求'注入 m0 首选失败后 full 臂让位仍 goal_reached'，runner 实现的判据是"
        " main_success_rate ≥ 2/3 —— 两者不同一，本模块按 runner 口径复算并如实登记该代理",
        "L4:冻结文本还要求 bundle digest 逐臂一致与越权检查，逐臂 payload 未携带 digest ⇒ 不可机检",
    ]
    return {
        "recomputed_lines": recomputed,
        "recorded_lines": recorded,
        "lines_match_report": recomputed == recorded,
        "recomputed_all_pass": all(recomputed.values()),
        "outcome_recorded": report.get("outcome"),
        "unverifiable_clauses": unverifiable,
        "per_arm": {
            name: {
                "main_success_rate": row.get("main_success_rate"),
                "total_steps_with_injection": row.get("total_steps_with_injection"),
                "executed_actions_with_injection": row.get("executed_actions_with_injection"),
                "unseen_success": row.get("unseen_success"),
            }
            for name, row in sorted(per_arm.items())
        },
    }


def run_unified_entry_live(*, wall_cap_seconds: float = UNIFIED_ENTRY_WALL_CAP) -> dict[str, Any]:
    """现场重跑统一入口证据包 ⇒ 07 §4.2 所要求的"输入到实际结果"展示。

    产物只写临时目录：**封存报告不覆写**（预注册 §4）。同一份 runner、同一套默认参数，
    所以这里复算出的 L1–L4 与封存件可比；差异本身就是读数（非确定性会在此暴露）。
    """

    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="cap0-f-unified-entry-") as tmp:
            scratch = Path(tmp) / "unified_entry_evidence_live.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(UNIFIED_ENTRY_RUNNER),
                    "--budget-approved",
                    "--report",
                    str(scratch),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=wall_cap_seconds,
            )
            elapsed = time.perf_counter() - started
            if not scratch.is_file():
                return {
                    "status": "not_executed",
                    "reason": "runner 未产出报告 ⇒ 不记通过",
                    "returncode": proc.returncode,
                    "stderr_tail": (proc.stderr or "").strip()[-400:],
                    "elapsed_seconds": round(elapsed, 3),
                }
            report = json.loads(scratch.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        return {
            "status": "not_executed",
            "reason": f"超出冻结 wall cap {wall_cap_seconds}s ⇒ 按停止线记录，不看结果调线",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except (OSError, ValueError) as exc:
        return {
            "status": "not_executed",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    adjudged = adjudicate_unified_entry(report)
    sealed: dict[str, Any] | None = None
    if UNIFIED_ENTRY_SEALED_REPORT.is_file():
        try:
            sealed = adjudicate_unified_entry(
                json.loads(UNIFIED_ENTRY_SEALED_REPORT.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError):
            sealed = None
    return {
        "status": "executed",
        "runner": _relative_path(UNIFIED_ENTRY_RUNNER),
        "task": report.get("task"),
        "wall_cap_seconds": wall_cap_seconds,
        "elapsed_seconds": round(elapsed, 3),
        "returncode": proc.returncode,
        **adjudged,
        "matches_sealed_report": (
            None if sealed is None else sealed["per_arm"] == adjudged["per_arm"]
        ),
    }


def _relative_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def f_dimension_gate(items: list[dict[str, Any]], live: dict[str, Any]) -> dict[str, Any]:
    """维度门：至少一项满足其独立冻结门，**且该项自己**展示输入到实际结果（评价集 gates.F）。

    现场重跑的 bundle 证据单独记在 `end_to_end_demonstration` 里：它证明的是统一执行入口，
    不代替某一项自己的能力链 —— 拿它把 F 维点亮是用代理信号冒充验收（07 §4.2 后半句）。
    """

    passing_gate = [row["id"] for row in items if row.get("gate_verdict") == "pass"]
    passing_with_chain = [row["id"] for row in items if row.get("verdict") == "pass"]
    shown = (
        live.get("status") == "executed"
        and live.get("recomputed_all_pass") is True
        and live.get("lines_match_report") is True
        and live.get("matches_sealed_report") is not False
    )
    return {
        "criterion": "至少一项满足其独立冻结门，并展示输入到实际结果（07 §4.2 / 评价集 gates.F）",
        "items_passing_frozen_gate": passing_gate,
        "items_passing_gate_and_must_show": passing_with_chain,
        "end_to_end_demonstration": {
            "unified_entry_reproduced": shown,
            "status": live.get("status"),
            "counts_as_item_chain": False,
        },
        "verdict": "pass" if passing_with_chain else "partial",
        "reason": (
            "" if passing_with_chain else "过冻结门的项均未机检其 must_show 链 ⇒ 维度不冒称通过"
        ),
    }


__all__ = [
    "adjudicate_f_items",
    "adjudicate_unified_entry",
    "f_dimension_gate",
    "run_unified_entry_live",
]
