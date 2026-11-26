"""14.8 package closeout and N15.1 relay proposal.

This stage does not train, does not build a model and does not read the sealed
confirm split.  It aggregates the 14.1-14.6 cards, records the corrected
14.6 negative exit, writes the package ledger and the root-hypothesis review,
and binds the N15.1 proposal (span-bound compositional reader + self-prefix
training) with its declared data, feedback, budget, gates and named exits.

The 14.7 confirmation was not released because the 14.6 task gate failed.  The
14.6 final candidates are evidence, not the N15.1 parent.  N15.1.0 is the only
next executable step and requires owner approval; N15.2 is conditional on the
N15.1.2 confirmation and is not approved here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DESIGN = "plans/reference/PLAN-N15-01_SPAN_BOUND_CONTINUOUS_20261102.md"
REPORT = "reports/taiji_core_record_relation_14_8_closeout_20261102.json"
MANIFEST = "plans/manifests/taiji_core_record_relation_14_8_closeout_20261102.json"
CLOSEOUT_SOURCE = "scripts/training/closeout_taiji_record_relation_14_8.py"

STAGE_CARDS = {
    "14.1": "reports/taiji_core_14_1_split_audit_20261102.json",
    "14.2": "reports/taiji_core_14_2_split_rerun_20261102.json",
    "14.3": "reports/taiji_core_read_source_14_3_20261102.json",
    "14.4": "reports/taiji_core_record_relation_14_4_20261102.json",
    "14.5": "reports/taiji_core_record_relation_14_5_20261102.json",
    "14.6.0": "reports/taiji_core_record_relation_14_6_preflight_20261102.json",
    "14.6": "reports/taiji_core_record_relation_14_6_20261102.json",
    "14.6.v1": "reports/taiji_core_record_relation_14_6_20261102_v1_engineering_failure.json",
}
STAGE_MANIFESTS = {
    "14.6": "plans/manifests/taiji_core_record_relation_14_6_20261102.json",
    "14.6.v1": (
        "plans/manifests/taiji_core_record_relation_14_6_20261102_v1_engineering_failure.json"
    ),
}

EXPECTED_STATUS = {
    "14.1": "split_defect_confirmed",
    "14.2": "goal_copy_shortcut_confirmed",
    "14.3": "pathway_unreachable",
    "14.4": "passed_record_relation_graph_engineering_only",
    "14.5": "priced_within_compare_cap",
    "14.6.0": "passed_14_6_checkpoint_preflight",
    "14.6": "native_byte_task_failed",
    "14.6.v1": "engineering_failure",
}

# Declared stage caps from the 14 package.  The 14.8 closeout is a read-only
# document/aggregation stage, so it declares a small cap and 0 Adam.
STAGE_CAPS = {
    "14.1": (None, None),
    "14.2": (40.0, 120.0),
    "14.3": (90.0, 240.0),
    "14.4": (600.0, 1200.0),
    "14.5": (600.0, 1200.0),
    "14.6.0": (300.0, 600.0),
    "14.6": (14400.0, 28800.0),
    "14.7": (None, None),
    "14.8": (300.0, 600.0),
}
STAGE_ADAM = {
    "14.1": 0,
    "14.2": 0,
    "14.3": 0,
    "14.4": 12,
    "14.5": 12,
    "14.6.0": 18,
    "14.6": 1800,
    "14.7": 0,
}
CLOSEOUT_CPU_SECONDS = 300
CLOSEOUT_WALL_SECONDS = 600
CLOSEOUT_PRODUCT_BYTES = 1 * 1024**3

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


class InstrumentError(RuntimeError):
    """A stage card, status or cost invariant failed."""


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(
    rows: list,
    check_id: str,
    passed: bool,
    observed,
    note: str = "",
    *,
    required: bool = True,
) -> None:
    rows.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "required": bool(required),
            "observed": observed,
            "note": note,
        }
    )


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"path": str(path), "sha256": sha256_text(path), "bytes": path.stat().st_size}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_json(relative_path: str) -> dict:
    path = ROOT / relative_path
    return cast(dict, json.loads(path.read_text(encoding="utf-8")))


def stage_cost(step: str, card: dict) -> tuple[float, float, int]:
    if step == "14.1":
        return 0.0, 0.0, 0
    if step == "14.3":
        return float(card["cpu_seconds"]), float(card["wall_seconds"]), 0
    if step in ("14.4", "14.5", "14.6.0", "14.6"):
        cost = card["cost"]
        cpu = float(cost["total_cpu_seconds"])
        wall = float(cost["wall_seconds"])
        adam = int(cost.get("adam_calls", STAGE_ADAM.get(step, 0)))
        return cpu, wall, adam
    if step == "14.2":
        cost = card["cost"]
        return float(cost["cpu_seconds"]), float(cost["wall_seconds"]), 0
    raise InstrumentError(f"unknown stage {step}")


def stage_within(cpu: float, wall: float, cap: tuple) -> bool:
    cap_cpu, cap_wall = cap
    if cap_cpu is None or cap_wall is None:
        return True
    return bool(cpu <= cap_cpu and wall <= cap_wall)


def run_type_check(paths: list[str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--follow-imports=skip",
        "--ignore-missing-imports",
        *paths,
    ]
    try:
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
        wall = time.perf_counter() - start
        errors = 0
        for line in result.stdout.splitlines():
            if line.startswith("Found ") and " error" in line:
                errors = int(line.split()[1])
        return {
            "returncode": result.returncode,
            "errors": errors,
            "output": result.stdout.strip(),
            "wall_seconds": wall,
            "passed": result.returncode == 0 and errors == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "returncode": None,
            "errors": None,
            "output": f"{type(exc).__name__}: {exc}",
            "wall_seconds": 0.0,
            "passed": False,
        }


def run_lint(paths: list[str]) -> dict:
    start = time.perf_counter()
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    ruff_format = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    wall = time.perf_counter() - start
    return {
        "ruff": {"returncode": ruff.returncode, "output": ruff.stdout.strip()},
        "ruff_format": {
            "returncode": ruff_format.returncode,
            "output": ruff_format.stdout.strip(),
        },
        "black": {
            "returncode": None,
            "status": "not_run",
            "reason": (
                "the installed black 26.5.1 does not terminate on the stage runners "
                "in this environment; ruff format is the declared formatter check"
            ),
        },
        "wall_seconds": wall,
        "passed": ruff.returncode == 0 and ruff_format.returncode == 0,
    }


def root_hypothesis_review(card14_6: dict) -> dict:
    task = card14_6["gates"]["task"]["observed"]
    content = card14_6["gates"]["content"]["conditions"]
    structure = card14_6["gates"]["structure"]["rows"]
    return {
        "supported": [
            (
                "内容通路必要：无来源/错误记录相对 A 真实固定 8 例各降 "
                f"{content['no_source']['drop']:.2f}/{content['wrong_record']['drop']:.2f}；"
                f"错误位置降 {content['wrong_position']['drop']:.2f}"
            ),
            (
                "记录条件行为部分成立：A_full 两 seed 总体 completed_action_exact "
                f"{task['total']:.4f}；未见组合 {task['unseen_combination']:.4f}"
            ),
            "记录反序与真实同分：读取对记录顺序不敏感",
            "checkpoint/独立恢复、父身份与 1,800/1,800 正式更新全部成立",
        ],
        "not_supported": [
            (
                "任务门未过：总体 "
                f"{task['total']:.4f}/0.80、未见实体 {task['unseen_entity']:.4f}/0.60、"
                f"真/置换双侧 {task['both_sides']:.4f}/0.60、反序 {task['reverse']:.4f}/0.95"
            ),
            (
                "结构增量不稳健：27031 A−B/A−C 下界 "
                f"{structure[0]['bootstrap']['lower_95']:.4f}/"
                f"{structure[1]['bootstrap']['lower_95']:.4f}；27032 A−C 下界 "
                f"{structure[3]['bootstrap']['lower_95']:.4f}"
            ),
            "retention_not_admissible：没有合格父能力集，不签旧能力保持",
            "连续状态、跨 session、反馈纠错、主动选择、生长均未检验",
        ],
        "architecture_attribution": [
            (
                "最弱轴是绑定而非“读不读”：未见实体 0.2125、真/置换双侧 0.4438；"
                "source top-1 仅 0.3125；池化摘要选记录不能稳定绑定新描述"
            ),
            (
                "记录级寻址 A−C 点估计 +0.106/+0.131 为正，但 8 world bootstrap 仅 1/4 过；"
                "局部编码 B_local 接近 A；不能整体归因“层级架构无效”，也不能签其必要增量"
            ),
            (
                "rescue 从 0.75 降到 0.00：当前读取→H→表达接口对一次正确内容注入不稳定，"
                "新候选须把 rescue 升为门"
            ),
        ],
        "data_training_attribution": [
            (
                "不是剂量不足：训练批次 NLL 约 1e-4，开发教师强制 0.8487，自由生成 0.5844；"
                "继续贴合教师强制前缀不会自动修好自由生成与未见实体"
            ),
            ("自由生成与教师强制有差：14.6 训练只用教师强制前缀；N15.1 加入声明的自前缀课程"),
            (
                "数据面已消费：14.6 dev 不再作为新证据；N15.1 用新的 24/12/12 v2 数据与新词表，"
                "旧 confirm 保持封存"
            ),
        ],
        "do_not_claim": [
            "general Chinese / free language",
            "L3/L4",
            "continuous internal life",
            "cross-session experience / correction / feedback",
            "active choice without prompt",
            "retention of a qualified old capability",
            "structural growth benefit",
        ],
    }


def coverage(card14_6: dict) -> dict:
    return {
        "native_chinese": (
            "partial: A_full two-seed completed_action_exact "
            f"{card14_6['gates']['task']['observed']['total']:.4f}; task gate failed"
        ),
        "continuous_state": "not tested; 14.6 uses per-example resets, not one continuous session",
        "plasticity": "parameters updated; no qualified retention set",
        "cross_session": "not run; no qualified parent capability",
        "feedback_correction": "not run; belongs to the conditional N15.2 contract",
        "active_choice": "not run",
        "growth": "not claimed; 09 negative evidence retained",
        "confirm": "14.6 confirm sealed and unread; N15.1 confirm to be sealed until N15.1.2",
        "general_chinese": False,
        "L3_L4": False,
    }


def next_design(design_sha: str) -> dict:
    return {
        "package": "N15.1",
        "design_doc": DESIGN,
        "design_doc_sha256": design_sha,
        "mechanism": "span-bound compositional reader + self-prefix training",
        "not_another_head": (
            "replaces the 14.4 pooled-record selection with a learned goal-description "
            "to record-description span alignment; keeps the same subject H/F, inherited "
            "encoder, generate/mix heads and recovery protocol"
        ),
        "data": (
            "taiji-composition-fixture-v2: 24 train / 12 dev / 12 confirm worlds, "
            "288/240/240 examples, new seeds 48021/48022 and order seeds 28021/28022, "
            "new five-object/five-place vocabulary; 14.6 dev retired, 14.6 confirm sealed"
        ),
        "feedback": (
            "raw 27-byte action plus EOS byte NLL and the declared self-prefix curriculum; "
            "no entity span, action id, host repair, gold record label or cross-session feedback"
        ),
        "stages": [
            {"id": "N15.1.0a", "name": "v2 data", "cpu": 60, "wall": 120, "adam": 0},
            {"id": "N15.1.0b", "name": "graph engineering", "cpu": 600, "wall": 1200, "adam": 12},
            {"id": "N15.1.0c", "name": "pricing", "cpu": 600, "wall": 1200, "adam": 12},
            {"id": "N15.1.0d", "name": "checkpoint preflight", "cpu": 300, "wall": 600, "adam": 18},
            {
                "id": "N15.1.1",
                "name": "formal comparison",
                "cpu": 18000,
                "wall": 36000,
                "adam": 1800,
                "product_bytes": 4 * 1024**3,
            },
            {
                "id": "N15.1.2",
                "name": "confirmation",
                "cpu": 2000,
                "wall": 4000,
                "adam": 0,
                "product_bytes": 1 * 1024**3,
            },
        ],
        "budget": {
            "formal_cpu_seconds": 18000,
            "formal_wall_seconds": 36000,
            "formal_adam_calls": 1800,
            "formal_product_bytes": 4 * 1024**3,
            "single_thread": True,
            "peak_ram_gib": 8,
        },
        "gates": {
            "task": {
                "total": 0.80,
                "unseen_combination": 0.60,
                "unseen_entity": 0.60,
                "both_sides": 0.60,
                "reverse": 0.95,
            },
            "structure": {"gain": 0.10, "bootstrap": "per-seed 8-world paired 2000, 95% lower > 0"},
            "span_binding": {"source_top1": 0.60},
            "content_path": {"drop": 0.10},
            "rescue": {"max_drop": 0.10},
            "retention": "retention_not_admissible when no qualified parent set",
            "confirm": "only after the N15.1.1 task gate",
        },
        "exits": [
            "engineering_failure",
            "instrument_invalid",
            "budget_not_admitted",
            "budget_stop",
            "native_byte_task_failed",
            "span_binding_component_gain_unresolved",
            "content_path_not_necessary",
            "span_intervention_unstable",
            "span_binding_supported_on_declared_task",
        ],
        "conditional_N15_2": (
            "only after the N15.1.2 confirmation on the same qualified parent; "
            "separate two-session feedback/correction/active-choice contract; "
            "does not auto-approve 12/13"
        ),
        "owner_approval_required": True,
        "next_executable_step": (
            "N15.1.0 data v2, then graph engineering, pricing and checkpoint preflight; "
            "no training before pricing and preflight"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--manifest", default=MANIFEST)
    args = parser.parse_args()
    report_path = ROOT / args.report
    manifest_path = ROOT / args.manifest
    if report_path.exists() or manifest_path.exists():
        raise FileExistsError("the 14.8 closeout cards are write-once")
    design_path = ROOT / DESIGN
    if not design_path.exists():
        raise InstrumentError("the N15.1 proposal is missing")
    design_text = design_path.read_text(encoding="utf-8")
    design_sha = sha256_text(design_path)
    closeout_sha = sha256_text(ROOT / CLOSEOUT_SOURCE)

    cards = {step: load_json(path) for step, path in STAGE_CARDS.items()}
    manifests = {step: load_json(path) for step, path in STAGE_MANIFESTS.items()}
    statuses = {step: card.get("status") for step, card in cards.items()}
    if statuses != EXPECTED_STATUS:
        raise InstrumentError(f"stage statuses drifted: {statuses}")

    card14_6 = cards["14.6"]
    if card14_6.get("passed") is not True or card14_6.get("failed_required"):
        raise InstrumentError("the corrected 14.6 card is not engineering-valid")
    if card14_6.get("release_14_7") is not False:
        raise InstrumentError("the 14.6 card released 14.7 despite the failed task gate")
    if card14_6.get("confirm_read") is not False:
        raise InstrumentError("the 14.6 card reports a confirm read")
    if card14_6["gates"]["task"]["passed"] is not False:
        raise InstrumentError("the 14.6 task gate is not the declared failure")
    if card14_6["gates"]["content"]["passed"] is not True:
        raise InstrumentError("the 14.6 content gate is not the declared pass")
    if card14_6["gates"]["structure"]["passed"] is not False:
        raise InstrumentError("the 14.6 structure gate is not the declared failure")
    v1 = cards["14.6.v1"]
    if v1.get("failed_required") != ["evaluation_complete"]:
        raise InstrumentError("the v1 failure card is not the declared evaluation_complete bug")
    correction = card14_6.get("correction") or {}
    if correction.get("v1_report_sha256") != sha256_text(ROOT / STAGE_CARDS["14.6.v1"]):
        raise InstrumentError("the corrected 14.6 card does not bind the v1 report")
    if correction.get("no_model_forward") is not True:
        raise InstrumentError("the corrected 14.6 card does not declare a no-forward rescore")
    if manifests["14.6"].get("status") != "native_byte_task_failed":
        raise InstrumentError("the corrected 14.6 manifest status drifted")
    if manifests["14.6"].get("release_14_7") is not False:
        raise InstrumentError("the corrected 14.6 manifest released 14.7")
    if card14_6.get("data", {}).get("digest") is None:
        raise InstrumentError("the 14.6 data digest is missing")

    if "14.7" in cards:
        raise InstrumentError("14.7 is not released but a card is present")

    # Stage cost ledger.  The 14.7 row is an explicit zero, not a missing card.
    stage_rows = []
    for step in ("14.1", "14.2", "14.3", "14.4", "14.5", "14.6.0", "14.6"):
        cpu, wall, adam = stage_cost(step, cards[step])
        cap = STAGE_CAPS[step]
        stage_ok = stage_within(cpu, wall, cap)
        stage_rows.append(
            {
                "step": step,
                "status": statuses[step],
                "card": STAGE_CARDS[step],
                "sha256": sha256_text(ROOT / STAGE_CARDS[step]),
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "adam_calls": adam,
                "cap_cpu_seconds": cap[0],
                "cap_wall_seconds": cap[1],
                "within": stage_ok,
            }
        )
    stage_rows.append(
        {
            "step": "14.7",
            "status": "not_released",
            "card": None,
            "sha256": None,
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
            "adam_calls": 0,
            "cap_cpu_seconds": None,
            "cap_wall_seconds": None,
            "within": True,
            "note": "14.6 task gate did not pass; the sealed confirm was not consumed",
        }
    )
    package_cpu = sum(row["cpu_seconds"] for row in stage_rows)
    package_wall = sum(row["wall_seconds"] for row in stage_rows)
    package_adam = sum(row["adam_calls"] for row in stage_rows)
    all_stage_within = all(row["within"] for row in stage_rows)
    cap_cpu_sum = sum(
        row["cap_cpu_seconds"] for row in stage_rows if row["cap_cpu_seconds"] is not None
    )
    cap_wall_sum = sum(
        row["cap_wall_seconds"] for row in stage_rows if row["cap_wall_seconds"] is not None
    )

    # The closeout is read-only.  The type/lint child wall is charged as a CPU
    # upper bound so the stage cost does not hide the formatter/type subprocesses.
    type_check = run_type_check([CLOSEOUT_SOURCE])
    lint = run_lint([CLOSEOUT_SOURCE])
    tool_wall = float(type_check.get("wall_seconds", 0.0)) + float(lint.get("wall_seconds", 0.0))
    closeout_cpu = (time.process_time() - STARTED_CPU) + tool_wall
    closeout_wall = time.perf_counter() - STARTED_WALL
    closeout_within = (
        closeout_cpu <= CLOSEOUT_CPU_SECONDS and closeout_wall <= CLOSEOUT_WALL_SECONDS
    )
    if not closeout_within:
        raise InstrumentError("the closeout stage exceeded its declared cap")
    stage_rows.append(
        {
            "step": "14.8",
            "status": "closed_negative_with_followup_design",
            "card": REPORT,
            "sha256": None,
            "cpu_seconds": closeout_cpu,
            "wall_seconds": closeout_wall,
            "adam_calls": 0,
            "cap_cpu_seconds": CLOSEOUT_CPU_SECONDS,
            "cap_wall_seconds": CLOSEOUT_WALL_SECONDS,
            "within": closeout_within,
            "note": "no model forward, no Adam, no weight change",
        }
    )
    total_cpu = package_cpu + closeout_cpu
    total_wall = package_wall + closeout_wall
    total_adam = package_adam

    required_design_strings = (
        "N15.1",
        "span-bound",
        "24 train / 12 dev / 12 confirm",
        "18,000 CPU/36,000 wall/1,800 Adam/4GiB",
        "N15.2",
        "唯一下一步",
    )
    design_strings = {text: text in design_text for text in required_design_strings}
    design_ok = all(design_strings.values())

    checks: list[dict] = []
    check(
        checks,
        "stage_cards",
        statuses == EXPECTED_STATUS,
        {
            "statuses": statuses,
            "v1_report": STAGE_CARDS["14.6.v1"],
            "v1_report_sha256": sha256_text(ROOT / STAGE_CARDS["14.6.v1"]),
            "corrected_report": STAGE_CARDS["14.6"],
            "corrected_report_sha256": sha256_text(ROOT / STAGE_CARDS["14.6"]),
        },
    )
    check(
        checks,
        "package_exit",
        card14_6["status"] == "native_byte_task_failed"
        and card14_6["gates"]["task"]["passed"] is False
        and card14_6["gates"]["content"]["passed"] is True
        and card14_6["gates"]["structure"]["passed"] is False
        and card14_6["release_14_7"] is False
        and "14.7" not in cards,
        {
            "package_exit": card14_6["status"],
            "14_7": "not_released",
            "task_gate": card14_6["gates"]["task"]["passed"],
            "content_gate": card14_6["gates"]["content"]["passed"],
            "structure_gate": card14_6["gates"]["structure"]["passed"],
        },
    )
    check(
        checks,
        "cost_ledger",
        all_stage_within and closeout_within and total_adam == package_adam,
        {
            "stages": stage_rows,
            "package_cpu_seconds": package_cpu,
            "package_wall_seconds": package_wall,
            "package_adam_calls": package_adam,
            "closeout_cpu_seconds": closeout_cpu,
            "closeout_wall_seconds": closeout_wall,
            "closeout_product_cap_bytes": CLOSEOUT_PRODUCT_BYTES,
            "closeout_product_note": "report and manifest JSON cards",
            "total_cpu_seconds": total_cpu,
            "total_wall_seconds": total_wall,
            "cap_cpu_sum": cap_cpu_sum,
            "cap_wall_sum": cap_wall_sum,
        },
    )
    check(
        checks,
        "design_document",
        design_ok,
        {
            "path": DESIGN,
            "sha256": design_sha,
            "required_strings": design_strings,
            "declares_no_training_yet": "未执行" in design_text,
            "declares_owner_approval": "待 owner 准入" in design_text,
        },
    )
    check(
        checks,
        "no_model_forward",
        True,
        {
            "model_forwards": 0,
            "adam_calls": 0,
            "parameter_changes": 0,
            "closeout_cpu_seconds": closeout_cpu,
            "closeout_wall_seconds": closeout_wall,
            "evidence_rows_rescored_by_14_6": card14_6["correction"]["rows_reconstructed"],
            "note": (
                "14.8 only reads cards and the proposal; the 14.6 rows were already "
                "re-scored by the 14.6 audit card"
            ),
        },
    )
    check(checks, "type_check", type_check["passed"], type_check)
    check(checks, "lint", lint["passed"], lint)

    review = root_hypothesis_review(card14_6)
    cov = coverage(card14_6)
    proposal = next_design(design_sha)
    cost = {
        "package_stages": stage_rows,
        "package_cpu_seconds": package_cpu,
        "package_wall_seconds": package_wall,
        "package_adam_calls": package_adam,
        "closeout_cpu_seconds": closeout_cpu,
        "closeout_wall_seconds": closeout_wall,
        "closeout_product_cap_bytes": CLOSEOUT_PRODUCT_BYTES,
        "closeout_product_note": "report and manifest JSON cards",
        "total_cpu_seconds": total_cpu,
        "total_wall_seconds": total_wall,
        "adam_calls": 0,
        "model_forwards": 0,
        "parameter_changes": 0,
    }
    within = {
        "stages": [
            {
                "step": row["step"],
                "within": row["within"],
                "cap_cpu_seconds": row["cap_cpu_seconds"],
                "cap_wall_seconds": row["cap_wall_seconds"],
            }
            for row in stage_rows
        ],
        "all_stage_caps_within": all_stage_within,
        "closeout_within": closeout_within,
        "package_cpu_seconds": package_cpu,
        "package_wall_seconds": package_wall,
        "package_cpu_cap_sum": cap_cpu_sum,
        "package_wall_cap_sum": cap_wall_sum,
        "package_within": package_cpu <= cap_cpu_sum and package_wall <= cap_wall_sum,
        "all_within": all_stage_within
        and closeout_within
        and package_cpu <= cap_cpu_sum
        and package_wall <= cap_wall_sum,
        "limits": {
            "closeout_cpu_seconds": CLOSEOUT_CPU_SECONDS,
            "closeout_wall_seconds": CLOSEOUT_WALL_SECONDS,
            "closeout_product_bytes": CLOSEOUT_PRODUCT_BYTES,
            "package_cpu_cap_sum": cap_cpu_sum,
            "package_wall_cap_sum": cap_wall_sum,
        },
    }

    source_hashes = {
        path: sha256_text(ROOT / path)
        for path in (*STAGE_CARDS.values(), *STAGE_MANIFESTS.values())
    }
    source_hashes[DESIGN] = design_sha
    source_hashes[CLOSEOUT_SOURCE] = closeout_sha

    report: dict[str, Any] = {
        "schema": "taiji-record-relation-14-8-closeout-v1",
        "step": "14.8",
        "date": "2026-11-02",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "closed_negative_with_followup_design",
        "passed": True,
        "package_exit": "native_byte_task_failed",
        "release_14_7": False,
        "owner_approval_required": True,
        "checks": checks,
        "decision": {
            "14_7": "not_released",
            "package_exit": "native_byte_task_failed",
            "continue_same_interface": False,
            "no_new_dose_seed_step": True,
            "next_package": "N15.1",
            "next_executable_step": "N15.1.0 data/graph/pricing/preflight",
        },
        "stage_ledger": stage_rows,
        "root_hypothesis_review": review,
        "coverage": cov,
        "next_design": proposal,
        "cost": cost,
        "within": within,
        "source_hashes": source_hashes,
        "confirm_read": False,
        "no_model_forward": True,
        "no_adam_call": True,
        "no_weight_change": True,
        "limits": (
            "Closeout and proposal only: 0 model forward, 0 Adam, 0 weight change; "
            "N15.1 requires owner approval; no general Chinese, L3/L4, continuous "
            "state, cross-session, growth, or active-choice claim; confirm sealed."
        ),
        "error": None,
        "type_check": type_check,
        "lint": lint,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    manifest_data: dict[str, Any] = {
        "schema": "taiji-record-relation-14-8-closeout-manifest-v1",
        "step": "14.8",
        "date": "2026-11-02",
        "package_stages": stage_rows,
        "package_exit": "native_byte_task_failed",
        "release_14_7": False,
        "root_hypothesis_review": review,
        "coverage": cov,
        "next_design": proposal,
        "decision": report["decision"],
        "cost": cost,
        "within": within,
        "source_hashes": source_hashes,
        "confirm_read": False,
        "next": "N15.1.0 data/graph/pricing/preflight after owner approval",
        "owner_approval_required": True,
    }
    manifest_info = write_json(manifest_path, manifest_data)
    report["manifest"] = {
        "path": relative(manifest_path),
        "sha256": manifest_info["sha256"],
        "bytes": manifest_info["bytes"],
    }
    check(
        checks,
        "manifest_identity",
        manifest_info["sha256"] == sha256_text(manifest_path),
        {"path": report["manifest"]["path"], "sha256": manifest_info["sha256"]},
    )
    required = [row for row in checks if row.get("required", True)]
    report["required_checks"] = len(required)
    report["failed_required"] = [row["id"] for row in required if not row["passed"]]
    report["passed"] = not report["failed_required"]
    if not report["passed"]:
        report["status"] = "engineering_failure"
    info = write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "package_exit": report["package_exit"],
                "release_14_7": report["release_14_7"],
                "failed_required": report["failed_required"],
                "next": report["decision"]["next_executable_step"],
                "owner_approval_required": report["owner_approval_required"],
                "package_cpu_seconds": package_cpu,
                "package_wall_seconds": package_wall,
                "closeout_cpu_seconds": closeout_cpu,
                "design": DESIGN,
                "report": info["path"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
