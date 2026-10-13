"""台账／队首／预注册里点名的 `reports/*.json`，到底有没有进过库？（DEBT-G134 的载体）

起因是本轮盘点时撞上的事实（㊵-775 ⑨／DEBT-G134）：台账 ㊵-565／567 引用的两枚 N2 乙档读数件
**从未提交过**（`git log -- <path>` 命中 0），而盘上那枚件与台账引用的数逐位相符——
所以"引用今天还站得住"与"仓内不可复算"可以同时为真。这类缺席只会随时间变贵，不会自己出现，
所以把它做成一台能红的普查仪，而不是做成我的记性。

取法刻意按**版本控制面**走（`git ls-files`），不在大树上递归扫盘——本仓记过 `rglob` 在这棵带
`node_modules` 与 `output/` 的树上会挂死（㊵-779 ④ 又实测了一次）。

三条判据都能为假：

* `citation_face_scanned`——三枚引用面（08 台账／03 队首／`plans/reference` 的在册件）必须都读到了非零字节；
  读不到就不是"没有引用"，而是扫描失败（fail-closed，红）。
* `every_cited_report_is_tracked`——点到的每一枚 `reports/*.json` 都必须在索引面里。
* `every_cited_report_exists_on_disk`——点到的每一枚还必须真在盘上（缺盘＝引用连原文都没有）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FORMAT = "taiji-ledger-cited-reports-tracked-v1"
CITE_RE = re.compile(r"reports/[A-Za-z0-9_.\-]+\.json")
READING_LIMIT = (
    "本件只回答一个问题：文档里点名的读数件，仓里到底有没有。它不判读数的对错，"
    "也不判引用是否恰当——一枚在册但已被更正作废的件照样算『在册』。"
)


def _tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "reports"], capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr.strip()[:200]}")
    return {line.strip() for line in result.stdout.split("\n") if line.strip()}


def _reference_docs() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "plans/reference"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr.strip()[:200]}")
    return [line.strip() for line in result.stdout.split("\n") if line.endswith(".md")]


def _citations(rel_paths: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for rel in rel_paths:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            continue
        text_ = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(CITE_RE, text_):
            name = f"reports/{match.group(0).split('/', 1)[1]}"
            found.setdefault(name, []).append(rel)
    return found


def _classify(cited: dict[str, list[str]], texts: dict[str, str]) -> dict[str, list[str]]:
    """点名的件分三桶——不拆开就会把"还没跑的输出"当成"丢了的证据"。

    * `malformed`＝写作 `reports/...json` 这种省略形，它不是件名，是**引用写法不合规**（该修文档）。
    * `planned_output`＝上文里出现 `--out-report`，即文档在说"将来落到这个件"，今天不存在是正常的。
    * `cited_as_reading`＝其余，＝被当作**已有读数**引用；这一桶才归本件判。
    """

    buckets: dict[str, list[str]] = {"malformed": [], "planned_output": [], "cited_as_reading": []}
    for name in sorted(cited):
        base = name.split("/", 1)[1]
        if base.startswith(".") or ".." in base:
            buckets["malformed"].append(name)
            continue
        contexts = []
        for rel in cited[name]:
            text_ = texts.get(rel, "")
            start = 0
            while True:
                at = text_.find(base, start)
                if at < 0:
                    break
                contexts.append(text_[max(0, at - 90) : at])
                start = at + 1
        if contexts and all("--out-report" in item for item in contexts):
            buckets["planned_output"].append(name)
        else:
            buckets["cited_as_reading"].append(name)
    return buckets


def build() -> dict[str, Any]:
    tracked = _tracked_files()
    faces = [
        "plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md",
        "plans/active/roadmap/03_CURRENT_EXECUTION.md",
    ]
    sizes = {}
    for rel in faces:
        path = PROJECT_ROOT / rel
        sizes[rel] = path.stat().st_size if path.is_file() else 0
    rel_paths = faces + _reference_docs()
    texts: dict[str, str] = {}
    for rel in rel_paths:
        path = PROJECT_ROOT / rel
        if path.is_file():
            texts[rel] = path.read_text(encoding="utf-8", errors="replace")
    cited = _citations(rel_paths)
    buckets = _classify(cited, texts)
    readings = buckets["cited_as_reading"]
    untracked = [name for name in readings if name not in tracked]
    missing_on_disk = [name for name in readings if not (PROJECT_ROOT / name).is_file()]
    flagged = sorted(set(untracked) | set(missing_on_disk))
    payload = {
        "format": REPORT_FORMAT,
        "citation_faces": {
            "ledger_and_execution": sizes,
            "reference_docs_scanned": len(_reference_docs()),
        },
        "tracked_reports_total": len(tracked),
        "cited_reports_total": len(cited),
        "buckets": {key: len(value) for key, value in buckets.items()},
        "cited_as_reading": readings,
        "planned_outputs": buckets["planned_output"],
        "malformed_citations": buckets["malformed"],
        "not_tracked": untracked,
        "missing_on_disk": missing_on_disk,
        "citations_per_report": {name: sorted(set(cited[name]))[:4] for name in flagged},
        "verdicts": {
            "citation_face_scanned": all(value > 0 for value in sizes.values()),
            "no_malformed_citation": not buckets["malformed"],
            "every_cited_report_is_tracked": not untracked,
            "every_cited_report_exists_on_disk": not missing_on_disk,
        },
        "reading_limit": READING_LIMIT,
    }
    #: 判定跟着读数一起成形（而不是只在 stdout 里算一次），这样被写下来的件自己就说得出红在哪条。
    payload["failing_verdicts"] = [key for key, value in payload["verdicts"].items() if not value]
    payload["verdict"] = "holds" if not payload["failing_verdicts"] else "violated"
    return payload


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="台账／预注册点名的读数件是否在册（DEBT-G134）")
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    out_path = args.out_report if args.out_report.is_absolute() else PROJECT_ROOT / args.out_report
    if out_path.exists() and not args.overwrite:
        print(
            json.dumps(
                {"format": REPORT_FORMAT, "status": "refuse_overwrite", "path": str(out_path)}
            )
        )
        return 2
    try:
        payload = build()
    except (RuntimeError, OSError) as error:
        print(json.dumps({"format": REPORT_FORMAT, "status": "refuse_scan", "detail": str(error)}))
        return 2
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0 if payload["verdict"] == "holds" else 1


if __name__ == "__main__":
    raise SystemExit(main())
