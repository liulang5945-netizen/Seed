"""PLAN-N1-03 §3 三条判据的执行体：把"承载表／映射表／恢复归属"从文字变成能红的读数。

为什么要这一台
--------------
㊵-701／㊵-706 已经把守卫 `G-N1c-2`（写-only 键）变成仪器，但 §3 的 `J-N1c-1`／`J-N1c-2`／
`J-N1c-4` 三条到今天仍**只有文字**——N1 出口交付第三项要的正是这张表的**可验收性**，
而"我读过觉得每格都填了"不是验收。

三条各自的取法（点名到判据原文，不新增阈值字面量）
--------------------------------------------------
* **J-N1c-1 五环都有承载点**：解析 §1 表每一行的反引号实名，逐个要求它**能在仓内解析**——
  要么是一枚存在的路径，要么在 tracked `.py` 里以字面形式出现（`git ls-files` 面）。
  任一环给不出可解析实名 ⇒ `chain_incomplete` 并点名是哪一环。
* **J-N1c-2 映射表逐格非空**：§2 表 Q/E/M/W/R × 四列（生产者／实际消费者／更新来源／恢复归属）
  ＝20 格，逐格要求非空且不是纯占位（`—`／`-`／`TBD`）。空格 ⇒ `table_incomplete` 并点名行列。
  **这一条只判"登记形式"**，不是能力证据（件里 `what_this_does_not_prove` 一字段就是这么写的）。
* **J-N1c-4 恢复归属可复核**：§2"恢复归属"列里点名的每个**键名**，必须能在
  (a) 任一 tracked `reports/*.json`（退出件）里现读到，或
  (b) 在 `taiji/`、`seed/` 与跑道器里作为**字典键／构造关键字参数**被写出去（本件自扫这三张面）。
  两处都取不到 ⇒ 该格记 `unverifiable_recovery_claim`；**"没读到"不许读成"不需要恢复"**。
  写侧为什么不复用 `audit_taiji_write_only_key_face`：那件只出版它自己那 13 枚键的 writer 计数，
  而"恢复归属"列点名的 `command_surface`／`config`／`n5_shadow_request` 一枚都不在里面（2026-10-11 现读）。

零算力：不加载检查点、不训练、不更新权重。落件行尾 LF，stdout 只打 ASCII。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREREG = "plans/reference/PLAN-N1-03_chain_and_roles_prereg_20261011.md"
REPORT_FORMAT = "taiji-n1-03-role-table-face-v1"
#: 契约册（`test_n1_06`）按文件路径加载本件、也被起成真子进程，脚本自己那层目录先上 `sys.path`，
#: 统一落盘助手才解析得到（同㊵-790／802 那条例外约定）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

BACKTICK = re.compile(r"`([^`]+)`")
PLACEHOLDERS = {"", "-", "—", "–", "TBD", "tbd", "N/A", "待填"}
ROLE_COLUMNS = ("生产者", "实际消费者", "更新来源", "恢复归属")
ROLES = ("Q", "E", "M", "W", "R")


def tracked_files(root: Path, pattern: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", pattern],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return out


def parse_tables(text: str) -> tuple[list[list[str]], list[list[str]]]:
    """返回 (§1 五环表的数据行, §2 五角色表的数据行)。"""

    lines = text.split("\n")
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in lines:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    ring_rows: list[list[str]] = []
    role_rows: list[list[str]] = []
    for table in tables:
        for row in table:
            if len(row) >= 4 and row[0].startswith(("①", "②", "③", "④", "⑤")):
                ring_rows.append(row)
                continue
            first_cell = row[0].split(" ")[0] if row else ""
            if len(row) >= 5 and first_cell in ROLES:
                role_rows.append([first_cell] + row[1:])
    return ring_rows, role_rows


RESOLVE_FACES = ("taiji", "seed", "scripts/training")


def resolve_name(
    root: Path, name: str, py_blob: str, ring: str, tracked_py: list[str]
) -> dict[str, object]:
    """把一枚"仓内实名"解析成几类之一，并**按强度分级**。

    强解析（算承载）：`as_file`（相对仓库根的路径存在）、`as_tracked_basename`（该文件名是某枚
    tracked `.py` 的尾部路径）、`as_module`（点号路径映射到真实 `.py`／包目录）、
    `as_identifier`（不含点号与方括号的裸名在 tracked 源码里出现）。
    弱解析（**只披露、不算承载**）：`as_attribute`（点号路径的模块部分不存在，只有**长度≥3 的末段**
    在源码里）、`as_json_path`（读数件路径，按首末段命中）。
    分级是必要的：§1 第④环写着 `taiji.kernel.state.motor_context`，而 `taiji/kernel/` 在仓内不存在；
    第一版我还漏了两处——`train_seed_corpus.py` 只靠末段 `py` 命中（任何文件都会命中的空判据）、
    `substrate.state.*` 靠**空末段**命中。三者都是"取法自己造出来的假通过"。
    """

    candidate = name.split("#")[0].split(":")[0].strip()
    kinds: list[str] = []
    strong = False
    if candidate:
        if (root / candidate).is_file():
            kinds.append("as_file")
            strong = True
        elif candidate.endswith(".py") and any(path.endswith(candidate) for path in tracked_py):
            kinds.append("as_tracked_basename")
            strong = True
        head = candidate.split("[")[0]
        if "." in head:
            dotted = head.replace(".", "/")
            leaf = head.rsplit(".", 1)[-1].strip("*.").strip()
            if (root / f"{dotted}.py").is_file() or (root / dotted).is_dir():
                kinds.append("as_module")
                strong = True
            elif len(leaf) >= 3 and leaf in py_blob:
                kinds.append("as_attribute")
        if "[" in candidate:
            segments = [s for s in re.split(r"\[[^\]]*\]|[.]", candidate) if s]
            if len(segments) >= 2 and segments[0] in py_blob and segments[-1] in py_blob:
                kinds.append("as_json_path")
        if not kinds and candidate in py_blob and "." not in candidate and "[" not in candidate:
            kinds.append("as_identifier")
            strong = True
    return {
        "name": name,
        "ring": ring,
        "kinds": kinds,
        "resolved": bool(kinds),
        "strong": strong,
    }


def payload_written(root: Path, key: str) -> list[str]:
    """该键是否被**写进信封／退出件**：在 `taiji/`、`seed/` 与跑道器里找它的字典键或构造关键字用法。

    判据原文是"能在**信封或退出件**里现读到"。信封由跑道器落盘（㊵-626 的 `command_surface`、
    ㊵-667 的 `n5_shadow_request` 都是它写的），所以只看 `taiji/`＋`seed/` 会把真实存在的自述键
    误判成"未取到"——这是我第一版在这两枚键上犯的scope错，写成函数而不是靠我记得。
    """

    hits: list[str] = []
    patterns = (f'"{key}"', f"'{key}'", f"{key}=")
    for face in RESOLVE_FACES:
        for rel in tracked_files(root, f"{face}/*.py"):
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            if any(pattern in text for pattern in patterns):
                hits.append(rel)
    return hits


def key_face_names(cell: str) -> list[str]:
    return [token for token in BACKTICK.findall(cell) if re.fullmatch(r"[a-z][a-z0-9_]{2,}", token)]


def judge(prereg_text: str, root: Path) -> dict[str, object]:
    ring_rows, role_rows = parse_tables(prereg_text)
    if not ring_rows or not role_rows:
        raise ValueError(
            f"表格解析为空（五环 {len(ring_rows)} 行／五角色 {len(role_rows)} 行）："
            "这台仪器今天什么都没测到，零命中不许读成「表是完整的」或「没有缺格」"
        )
    tracked_py = tracked_files(root, "*.py")
    py_blob = "\n".join(
        (root / rel).read_text(encoding="utf-8", errors="replace") for rel in tracked_py
    )
    rings = []
    for row in ring_rows:
        names = BACKTICK.findall(" ".join(row[1:]))
        checked = [resolve_name(root, name, py_blob, row[0], tracked_py) for name in names]
        strong = sum(1 for item in checked if item["strong"])
        rings.append(
            {
                "ring": row[0],
                "named_anchors": len(checked),
                "strongly_resolved": strong,
                "weak_only_anchors": [
                    f"{item['name']}({'|'.join(item['kinds'])})"
                    for item in checked
                    if item["resolved"] and not item["strong"]
                ],
                "satisfied": strong >= 1,
                "unresolved": [item["name"] for item in checked if not item["resolved"]],
                "no_anchor_at_all": not checked,
            }
        )
    unresolved_rings = [r["ring"] for r in rings if not r["satisfied"]]
    dead_anchors = sorted({anchor for r in rings for anchor in r["unresolved"]})
    weak_anchors = sorted({anchor for r in rings for anchor in r["weak_only_anchors"]})
    verdict_1 = "chain_complete" if not unresolved_rings else "chain_incomplete"

    cells: list[dict[str, object]] = []
    seen_roles: list[str] = []
    for row in role_rows:
        seen_roles.append(row[0])
        for offset, column in enumerate(ROLE_COLUMNS, start=1):
            body = row[offset] if offset < len(row) else ""
            cells.append(
                {
                    "role": row[0],
                    "column": column,
                    "empty": body.strip() in PLACEHOLDERS,
                    "length": len(body),
                }
            )
    empty_cells = [f"{c['role']}/{c['column']}" for c in cells if c["empty"]]
    missing_roles = [role for role in ROLES if role not in seen_roles]
    verdict_2 = (
        "table_registered"
        if not empty_cells and not missing_roles and len(cells) == len(ROLES) * len(ROLE_COLUMNS)
        else "table_incomplete"
    )

    recovery_cells = {
        row[0]: (row[4] if len(row) > 4 else "") for row in role_rows if row[0] in ROLES
    }
    keys = sorted({token for cell in recovery_cells.values() for token in key_face_names(cell)})
    sealed_reports = {
        rel: (root / rel).read_text(encoding="utf-8", errors="replace")
        for rel in tracked_files(root, "reports/*.json")
    }
    recovery = []
    for key in keys:
        in_exit_artifact = any(key in blob for blob in sealed_reports.values())
        written_faces = payload_written(root, key)
        recovery.append(
            {
                "key": key,
                "in_sealed_report": in_exit_artifact,
                "written_into_envelope_by": written_faces,
                "verifiable": in_exit_artifact or bool(written_faces),
            }
        )
    unverifiable = [item["key"] for item in recovery if not item["verifiable"]]
    verdict_4 = (
        "recovery_claims_verifiable"
        if (keys and not unverifiable)
        else ("unverifiable_recovery_claim" if keys else "recovery_column_names_no_keys")
    )

    return {
        "format": REPORT_FORMAT,
        "prereg": PREREG,
        "j_n1c_1": {
            "verdict": verdict_1,
            "rings": rings,
            "rings_parsed": len(rings),
            "rings_without_evidence": unresolved_rings,
            "anchors_that_do_not_resolve": dead_anchors,
            "anchors_with_weak_resolution_only": weak_anchors,
            "ring_rule": "至少一枚**强**解析实名即算该环有承载；弱解析（模块路径不存在、只末段命中）"
            "单列成清单，不算承载也不算否证",
        },
        "j_n1c_2": {
            "verdict": verdict_2,
            "cells_expected": len(ROLES) * len(ROLE_COLUMNS),
            "cells_parsed": len(cells),
            "roles_parsed": seen_roles,
            "roles_missing": missing_roles,
            "empty_cells": empty_cells,
        },
        "j_n1c_4": {
            "verdict": verdict_4,
            "keys_named_in_recovery_column": keys,
            "write_side_scan_faces": list(RESOLVE_FACES),
            "checks": recovery,
            "unverifiable": unverifiable,
        },
        "what_this_does_not_prove": [
            "J-N1c-2 判的是「这张表存在且逐格非空」＝登记形式，不是能力证据；填得满不等于说得对",
            "J-N1c-1 的「实名可解析」只证明那个文件／键名在仓内存在，不证明它真的承载了对应那一环",
            "J-N1c-4 走的是「退出件里现读到」或「产品码把它写成 payload 键」两条道；"
            "两者都不成立时判 unverifiable，而不是「该键不需要恢复」",
        ],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_report(root: Path) -> dict[str, object]:
    prereg_text = (root / PREREG).read_text(encoding="utf-8")
    return judge(prereg_text, root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-report", default="reports/taiji_n1_03_role_table_face_20261011.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(Path(args.root))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = Path(args.root) / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "j_n1c_1": report["j_n1c_1"]["verdict"],
                "j_n1c_2": report["j_n1c_2"]["verdict"],
                "j_n1c_4": report["j_n1c_4"]["verdict"],
                "empty_cells": report["j_n1c_2"]["empty_cells"],
                "unverifiable": report["j_n1c_4"]["unverifiable"],
                "out": out.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
