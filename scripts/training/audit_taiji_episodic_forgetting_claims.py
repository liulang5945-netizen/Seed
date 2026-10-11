"""情景记忆"遗忘"断言的现扫仪器（DEBT-G98/G99 配套，J-N2f-4 的测量式改版）。

**为什么改版**：原判据写的是"现扫 `automatic_forgetting`／『自动遗忘』字样，命中即判 `forbidden_claim`"。
这个取法在两天里自指命中两次：第一次命中 VISION §5.3 的禁令原文，第二次命中我自己写的禁令件
`PLAN-N2-06`（它抄写"自动遗忘"来禁止它）。⇒ 任何"把违禁词抄进禁令"的文档都会红，
排除某一份文档的写法根本不成立。

**改版后的判据意图一字不动**（仍禁止"声称已具备遗忘能力"），只把测量式从"扫字样"换成"扫肯定式"：
命中要求同时满足 ①匹配肯定式；②同句不含否定标记（不能/不得/不许/禁止/不可/unsupported/absent/forbidden/
never/no longer）。正例与反例各配一支测，防止把"取法变宽"当成"问题修好"。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAT = "taiji-episodic-forgetting-claims-v1"

AFFIRMATIVE = (
    re.compile(r"会自动遗忘|能够自动遗忘|支持自动遗忘|已自动遗忘"),
    re.compile(r"已经遗忘(了)?(该|这条|该条|过期)"),
    re.compile(r"automatic_forgetting[^.]{0,40}(supported|enabled|implemented)"),
)
NEGATION = re.compile(
    r"不能|不得|不许|禁止|不可|不应|不承诺|未实现|unsupported|not supported|absent|forbidden|never|no longer"
)

DEFAULT_FACES = ("reports", "plans")


def scan(root: Path, faces: tuple[str, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scanned = 0
    for face in faces:
        for path in sorted((root / face).rglob("*.md")) + sorted((root / face).rglob("*.json")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            for number, line in enumerate(text.split("\n"), 1):
                if not any(pattern.search(line) for pattern in AFFIRMATIVE):
                    continue
                if NEGATION.search(line):
                    continue  # 禁令本体（含否定标记）不是断言
                rows.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "line": number,
                        "excerpt": line.strip()[:160],
                    }
                )
    return {
        "format": FORMAT,
        "status": "ok",
        "files_scanned": scanned,
        "faces": list(faces),
        "violations": rows,
        "violation_count": len(rows),
        "verdicts": {"no_affirmative_forgetting_claims": not rows},
        "reading_limit": (
            "本器只扫『肯定式声称已具备自动遗忘能力』。禁令本体（同句含否定标记）不计命中；"
            "字样出现在中性叙述里也不算——判据要拦的是断言，不是词。"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="scan for affirmative automatic-forgetting claims")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--face", action="append", dest="faces", help="repeatable; default reports+plans"
    )
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args(argv)
    faces = tuple(args.faces or DEFAULT_FACES)
    payload = scan(args.root, faces)
    if payload["files_scanned"] == 0:
        print(json.dumps({"status": "scan_face_empty", "faces": list(faces)}, ensure_ascii=True))
        return 2
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "files_scanned": payload["files_scanned"],
                "violation_count": payload["violation_count"],
                "verdicts": payload["verdicts"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
