"""对 `output/` `outputs/` `reports/` 的**未跟踪内容**做可删性分级（只读）。

**分级判据＝「删掉之后要不要重跑才能重新拿到」**：
- `zero_rebuild`（零重建）：**不需要任何重跑**就能重新得到，或本来就是可再生的中间件。
- `cpu_rebuild`（CPU 重建）：CPU 可重跑得到（数分钟到数小时量级）。
- `gpu_rebuild`（GPU 重建）：需GPU 才能得到。

**为什么以「要不要重跑」为唯一主判据**：这是本仓已定的判据切法
（已蒸馏的 `REPO_CLEANUP_ASSESSMENT_20260927` 用的是「零重建／CPU+网络／GPU」三档），
本轮只是把它落到当前 HEAD 上重新对账 —— **沿用既有判据，不发明新的**。

**已跟踪的文件不在本工具的口径内**：它们是冻结判据与读数证据
（`reports/` 1658 个 json ＝219 MB），按「冻结判据/证据只追加」不删。

用法：
    python scripts/training/classify_untracked_artifacts.py            # 人读报告
    python scripts/training/classify_untracked_artifacts.py --json     # 机器可读
退出码恒为 0（本工具只读、不删任何东西；「有没有该删的」是owner 决策）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("output", "outputs", "reports")

#: 三档定义（沿用已蒸馏评估的切法）
TIERS = {
    "zero_rebuild": "删后无需任何重跑（可再生中间件／日志／工具副产物）",
    "cpu_rebuild": "删后 CPU 可重跑得到（数分钟到数小时量级）",
    "gpu_rebuild": "删后需GPU 才能得到",
    "evidence": "刻意保留的证据（训练中间态 checkpoint）——**不建议删**",
}

#: 可再生中间件后缀：这些随时能由程序再生成，不需要保留。
REGENERABLE_SUFFIX = (".log", ".history", ".tmp", ".lock", ".pid", ".pyc")

#: **训练中间态checkpoint 目录**：单列为 `evidence` 档，**任何档都不建议删**。
#: 理由（本仓的既有实践，不是本工具自创）：`train_seed_corpus.py:880` 有
#: `--keep-checkpoints` 开关，而台账里记着该实验以 `keep-checkpoints on` 主动开启
#: 并与 `off` 档做过报价量级对比 ⇒ 这些中间态**是为「任何指标事后补算」刻意保留的**，
#: 不是训练留下的垃圾。若把 `.history` 一律按「可再生中间件」删掉，
#: 会静默毁掉"事后补算指标"这条能力。
EVIDENCE_DIR_MARKER = ".pt.history"

#: 明确是「证据」的东西，**任何档都不建议删**（即便未被跟踪）。
EVIDENCE_SUFFIX = (".json", ".md", ".txt", ".csv")

#: 单文件体积上限（KB）：超过此值一律至少 CPU 档。
BIG_KB = 10240


def dir_size_kb(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total // 1024


def tracked_set() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *ROOTS],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def classify_dir(root: Path, tracked: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.is_symlink():
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel in tracked:
            continue
        size_kb = max(1, path.stat().st_size // 1024)
        suffix = path.suffix.lower()
        # ── 判据：从「最不该删」往下走，每命中一条即定档
        if EVIDENCE_DIR_MARKER in path.as_posix():
            tier, why = (
                "evidence",
                "训练中间态 checkpoint（keep-checkpoints 刻意保留，供指标事后补算）",
            )
        elif suffix in REGENERABLE_SUFFIX:
            tier, why = "zero_rebuild", "可再生中间件/日志类"
        elif suffix in EVIDENCE_SUFFIX:
            tier, why = "cpu_rebuild", f"{suffix} 证据文件，未跟踪但仍是读数载体"
        elif suffix == ".pt":
            tier, why = "gpu_rebuild", "模型权重，重跑需 GPU"
        elif size_kb >= BIG_KB:
            tier, why = "gpu_rebuild", f"单文件 {size_kb} KB 的大体量产物"
        else:
            tier, why = "cpu_rebuild", "小体积产物，CPU 可重跑"
        rows.append(
            {
                "path": rel,
                "size_kb": size_kb,
                "tier": tier,
                "why": why,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tracked = tracked_set()
    all_rows: list[dict[str, object]] = []
    for name in ROOTS:
        root = REPO / name
        if root.is_dir():
            all_rows.extend(classify_dir(root, tracked))

    by_tier: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        by_tier[str(row["tier"])].append(row)

    if args.json:
        print(
            json.dumps(
                {
                    "tracked_in_roots": len(tracked),
                    "untracked_files": len(all_rows),
                    "tiers": {tier: len(rows) for tier, rows in by_tier.items()},
                    "rows": sorted(all_rows, key=lambda r: (-int(r["size_kb"]), str(r["path"]))),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    total_kb = sum(int(r["size_kb"]) for r in all_rows)
    print(f"已跟踪文件 {len(tracked)} 个（**不在本工具口径内，冻结证据不删**）")
    print(f"未跟踪文件 {len(all_rows)} 个，合计 ≈{total_kb:,} KB（{total_kb/1024/1024:.2f} GB）\n")
    for tier, desc in TIERS.items():
        rows = by_tier.get(tier, [])
        size = sum(int(r["size_kb"]) for r in rows)
        print(f"== {tier}：{len(rows)} 个，≈{size:,} KB（{size/1024/1024:.2f} GB）")
        print(f"   判据：{desc}")
        rows.sort(key=lambda r: -int(r["size_kb"]))
        for row in rows[:12]:
            print(f"     {int(row['size_kb']):>8,} KB  {row['path']}")
        if len(rows) > 12:
            print(f"     …另有 {len(rows) - 12} 个（`--json` 看全量）")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
