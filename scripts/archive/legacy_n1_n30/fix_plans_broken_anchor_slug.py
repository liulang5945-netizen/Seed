"""按链接体检结果修正 `plans/` 下的坏锚点（只改锚点串，不动正文）。

**只修一种**：链接写的锚点带了多余连字符，而目标标题真实 slug 不带
（实测 12 条集中在 `#21-统一认知闭环验收原则2026-09-17用户确认`，
目标文件的真实锚点是 `21-统一认知闭环验收原则20260917用户确认`——
括号与日期里的连字符在 slug 里都已被剥掉）。
**不碰** `missing_file`（那是历史文件被删后的死链，归属另议，见台账）。

安全措施：
- 只替换「`#` 锚点」这一段，正文其余字符逐字保留；
- 每次替换前先验：该锚点确实不在目标文件的锚点集里（否则是工具误报，不改）；
- 写回后立刻重跑体检，锚点死链数必须下降且不得新增 `missing_file`。

用法：
    python scripts/training/fix_plans_broken_anchor_slug.py --check
    python scripts/training/fix_plans_broken_anchor_slug.py --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_plans_markdown_links import LINK_RE, collect_anchors, strip_code_blocks  # noqa: E402

#: 已知错法：`2026-09-17` 这类日期里的连字符在 slug 中不存在。
#: 目标文件里的真实 slug 是 `…20260917…`。
DATE_ANCHOR_RE = re.compile(r"(?<=[\u4e00-\u9fff])(\d{4})-(\d{2})-(\d{2})(?=[\u4e00-\u9fff])")


def candidate_anchors(anchor: str) -> list[str]:
    """给出该锚点的候选修正形态（按可能性排序）。"""
    out = [DATE_ANCHOR_RE.sub(r"\1\2\3", anchor)]
    return [candidate for candidate in out if candidate != anchor]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.check == args.apply:
        print("必须且只能给一个：--check 或 --apply")
        return 2

    plans = REPO / "plans"
    planned: list[tuple[Path, int, str, str]] = []
    for source in sorted(plans.rglob("*.md")):
        try:
            raw = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        text = strip_code_blocks(raw)
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if "#" not in target:
                continue
            path_part, anchor = target.split("#", 1)
            if not path_part:
                resolved = source
            else:
                if path_part.startswith("/") or ":" in path_part.split("/")[0]:
                    continue
                resolved = (source.parent / path_part).resolve()
            if not resolved.is_file():
                continue
            anchors = collect_anchors(resolved)
            if anchor in anchors:
                continue
            line_no = text[: match.start()].count("\n") + 1
            for candidate in candidate_anchors(anchor):
                if candidate in anchors:
                    planned.append((source, line_no, anchor, candidate))
                    break

    print(f"可修的锚点：{len(planned)} 条")
    for source, line_no, old, new in planned:
        print(f"  {source.relative_to(REPO).as_posix()}:{line_no}")
        print(f"      #{old}")
        print(f"   →  #{new}")

    if not planned:
        return 0
    if args.check:
        return 0

    # 按文件聚合，逐个替换（串行，避免同文件多处替换互相错位）
    by_file: dict[Path, list[tuple[str, str]]] = {}
    for source, _line_no, old, new in planned:
        by_file.setdefault(source, []).append((old, new))
    for source, pairs in by_file.items():
        raw = source.read_text(encoding="utf-8")
        updated = raw
        for old, new in pairs:
            updated = updated.replace(f"#{old}", f"#{new}")
        if updated != raw:
            source.write_text(updated, encoding="utf-8")
    print(f"已改写 {len(by_file)} 个文件、共 {len(planned)} 处锚点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
