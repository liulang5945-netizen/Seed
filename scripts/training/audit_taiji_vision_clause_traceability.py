"""VISION 设计条款对账表的**可核验性**普查（DEBT-G97，静态、零算力、只读）。

**为什么要有它**：09 §1.3 的对账规则写着"每个 N 项落 PLAN-Nxx 时，必须消化表中归属该 N 项的全部条款
（逐条在 PLAN 内展开为实现设计，**缺条即不完整**）"。这句话要成立，就必须有一个**机械能答**的入口：
"V-18 这条现在由哪份件承载"。本器就是那个入口，它把"缺条即不完整"从一句靠人读的承诺
变成一条能复算的读数（同族的先例＝DEBT-G91 的封存件普查、DEBT-G92 的逐张量差）。

**两把尺并列出，不合并**（本仓纪律：一条独立读数会骗人）：

* `id_citations`：该条款号（`V-07` 这种字样，含 `V-08/11/13` 紧凑写法）在归属那份 N 的
  `PLAN-Nx-*.md` 里出现的次数——这是**身份级**可核验；
* `section_citations`：该条款登记的 VISION 章节锚（`§3.1` 这种）在同批文件里出现的次数——
  这是**锚点级**可核验，粒度更粗（一节可以装多条）。

**不许的读法**（写进 `reading_limit`，也写在这里）：本器量的是**可追溯性，不是语义消化**。
一条条款可能已经在件里被完整展开成实现设计、但作者没写条款号也没写章节锚 ⇒ 那会被数成
`neither`，这是"没有核对入口"而不是"没做"。反过来，命中章节锚也**不等于**那条已被展开。
所以 `neither` 的正确处置是"补登记或补展开"，不是"宣布失败"。

归属为**全局／出口后／备选**的条款不计入"必须现在可核验"的那一格（对账规则只对 N 项落件强制），
但它们照样逐行出版，`global_unmapped` 那类不许被省略掉。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_DOC = PROJECT_ROOT / "plans" / "active" / "roadmap" / "09_NEXT_MAINLINE_PLAN.md"
REFERENCE_DIR = PROJECT_ROOT / "plans" / "reference"
FORMAT = "taiji-vision-clause-traceability-v1"
CLAUSE_ID = re.compile(r"V-(\d{2})")
N_ATTRIBUTION = re.compile(r"N([1-6])")
SECTION_ANCHOR = re.compile(r"§\s*([0-9]+(?:\.[0-9]+)?[a-z]?)")
TABLE_ROW = re.compile(r"^\|\s*V-\d{2}\s*\|")
#: 归属里出现 `N5` 不等于"现在必须可核验"：`备选（N5 后若行动线开启再评）` 与
#: `出口后（…）` 都是**将来或备选**，把它们算进分母就是用被条件在结果上的集合去要覆盖率
#: （对账规则原文只强制"每个 N 项落 PLAN-Nxx 时"）。本行是我自己第一版的错，实测过才看得见。
NOT_REQUIRED_NOW = ("备选", "出口后", "不预支")


def _is_required_now(attribution: str) -> bool:
    if any(marker in attribution for marker in NOT_REQUIRED_NOW):
        return False
    return bool(N_ATTRIBUTION.search(attribution))


def _parse_table(doc_text: str) -> list[dict[str, Any]]:
    """从 09 §1.3 的表里逐行取「条款／章节锚／归属」，按竖线切，不按空格切。"""

    rows: list[dict[str, Any]] = []
    for line in doc_text.split("\n"):
        if not TABLE_ROW.match(line.strip()):
            continue
        cells = [one.strip() for one in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            raise ValueError(f"对账表行竖线不足（{len(cells)}）：{line[:80]!r}")
        ids = CLAUSE_ID.findall(cells[0])
        if not ids:
            raise ValueError(f"对账表首格没有条款号：{cells[0]!r}")
        sections = SECTION_ANCHOR.findall(cells[2]) if cells[2] else []
        rows.append(
            {
                "clause": f"V-{ids[0]}",
                "keywords": cells[1],
                "vision_section": sections[0] if sections else None,
                "attribution": cells[3],
                "carrier": cells[4],
                "n_codes": sorted({f"N{one}" for one in N_ATTRIBUTION.findall(cells[3])}),
                "required_now": _is_required_now(cells[3]),
            }
        )
    if not rows:
        raise ValueError(f"对账表为空：在 {doc_text.count(chr(10))} 行里没有一行以 `| V-nn |` 开头")
    return rows


def _plan_files(reference_dir: Path, n_code: str) -> list[Path]:
    return sorted(reference_dir.glob(f"PLAN-{n_code}-*.md"))


def _citation_counts(texts: list[str], clause: str, section: str | None) -> tuple[int, int]:
    """紧凑写法 `V-08/11/13` 也算命中：先按完整号找，再按 `V-08/…/13` 那一段里的裸两位数找。"""

    number = clause.split("-")[1]
    alt = str(int(number))
    id_hits = 0
    section_hits = 0
    for text in texts:
        id_hits += len(re.findall(re.escape(clause), text))
        # 紧凑列表：`V-08/11/13` 里 13 这条不会以 `V-13` 出现
        id_hits += len(re.findall(r"V-\d{2}(?:/\d{2})*/" + re.escape(alt) + r"(?![0-9])", text))
        if section:
            section_hits += len(re.findall(re.escape("§" + section), text))
    return id_hits, section_hits


def audit(doc_path: Path, reference_dir: Path = REFERENCE_DIR) -> dict[str, Any]:
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace")
    rows = _parse_table(doc_text)
    required = [one for one in rows if one["required_now"]]
    detail: list[dict[str, Any]] = []
    for row in rows:
        files = [f for code in row["n_codes"] for f in _plan_files(reference_dir, code)]
        texts = [f.read_text(encoding="utf-8", errors="replace") for f in files]
        id_hits, section_hits = _citation_counts(texts, row["clause"], row["vision_section"])
        detail.append(
            {
                **row,
                "plan_files_scanned": len(files),
                "id_citations": id_hits,
                "section_citations": section_hits,
                "traceability": (
                    "id"
                    if id_hits
                    else (
                        "section_only"
                        if section_hits
                        else ("unmapped" if row["required_now"] else "n/a")
                    )
                ),
            }
        )
    required_ids = {one["clause"] for one in required}
    id_cited = {one["clause"] for one in detail if one["id_citations"]}
    section_cited = {one["clause"] for one in detail if one["section_citations"]}
    neither = sorted(required_ids - id_cited - section_cited)
    return {
        "format": FORMAT,
        "status": "ok",
        "doc": str(doc_path),
        "reference_dir": str(reference_dir),
        "clause_rows_parsed": len(rows),
        "clauses_required_now": len(required_ids),
        "clauses_with_id_citation": len(required_ids & id_cited),
        "clauses_with_section_citation": len(required_ids & section_cited),
        "clauses_unmapped_both": len(neither),
        "unmapped_both_list": neither,
        "clauses_global_or_post_or_optional": len(rows) - len(required),
        "verdicts": {
            "table_parsed_every_row_has_clause_id": all(one["clause"] for one in detail),
            "every_required_clause_has_any_citation": not neither,
            "every_clause_has_id_level_citation": not (required_ids - id_cited),
        },
        "rows": detail,
        "reading_limit": (
            "this measures TRACEABILITY, not semantic digestion: a clause may already be fully "
            "expanded in a PLAN file by an author who wrote neither the V-id nor the section "
            "anchor, and that counts as unmapped (the correct action is to register or expand, "
            "not to declare failure). Conversely a section-anchor hit does NOT prove the clause "
            "was expanded, because one VISION section can carry several clauses. Worse, the "
            "anchor string is AMBIGUOUS ACROSS DOCUMENTS: `S5.1` inside 07_MINI_MODEL_DELIVERY "
            "is that doc's own CAP-0 section, not VISION 5.1, so section_citations is an upper "
            "bound and any claim built on it must name the citing sentence (measured 2026-10-11: "
            "the seven unmapped clauses' anchors hit 1-8 other docs, several of them false). "
            "Counts are taken over the PLAN files attributed to that clause's N-code only, so a "
            "clause discussed in another line's file is not credited here."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="VISION clause traceability census (read-only)")
    parser.add_argument("--doc", type=Path, default=PLAN_DOC, help="roadmap doc holding the table")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=REFERENCE_DIR,
        help="directory holding PLAN-Nx-*.md files",
    )
    parser.add_argument("--out-report", type=Path, required=True, help="where to write the JSON")
    args = parser.parse_args(argv)

    try:
        payload = audit(args.doc, args.reference_dir)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "census_failed", "error": str(error)}, ensure_ascii=True))
        return 2
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "clause_rows_parsed": payload["clause_rows_parsed"],
                "clauses_required_now": payload["clauses_required_now"],
                "clauses_with_id_citation": payload["clauses_with_id_citation"],
                "clauses_with_section_citation": payload["clauses_with_section_citation"],
                "clauses_unmapped_both": payload["clauses_unmapped_both"],
                "verdicts": payload["verdicts"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
