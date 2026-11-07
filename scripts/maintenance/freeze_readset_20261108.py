"""Freeze the read-set of the running training window (2026-11-08).

Static AST import closure only: this script imports nothing from ``taiji`` and
never builds a graph, loads a checkpoint or steps an optimizer.  It resolves the
local import graph of the runner plus the assets the runner reads, then writes:

* ``plans/manifests/taiji_frozen_readset_20261108.json`` (machine list, sha256)
* ``plans/reference/FROZEN-READSET_20261108.md`` (human table)

The point is a "do not touch" list: while these bytes are unchanged, editing any
other file cannot change what the running process already imported, and a
resumed run re-checks the same identity inputs (prereg sha, runner sha,
checkpoint sha) that this list pins.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/training/run_taiji_training_window_v2_20261108.py"
ASSETS = (
    (ROOT / "plans/manifests/taiji_training_window_v2_prereg_20261108.json", "pre-registration"),
    (
        ROOT / "plans/manifests/taiji_package20_4c_r2_freeze_20261105.json",
        "20.4c r2 freeze manifest",
    ),
    (
        ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt",
        "parent checkpoint (loaded by r2)",
    ),
)
MANIFEST = ROOT / "plans/manifests/taiji_frozen_readset_20261108.json"
DOC = ROOT / "plans/reference/FROZEN-READSET_20261108.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_module_file(name: str) -> Path | None:
    parts = name.split(".")
    candidates: list[Path] = []
    if parts[0] == "taiji" or (len(parts) > 1 and parts[:2] == ["scripts", "training"]):
        base = ROOT.joinpath(*parts)
        candidates += [base.with_suffix(".py"), base / "__init__.py"]
    elif len(parts) == 1:
        base = ROOT / "scripts" / "training" / parts[0]
        candidates += [base.with_suffix(".py")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def closure(seed: Path) -> tuple[list[Path], list[str]]:
    seen: dict[Path, None] = {}
    queue = [seed]
    missing: list[str] = []
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen[path] = None
        for name in sorted(imported_modules(path)):
            target = local_module_file(name)
            if target is not None:
                queue.append(target)
            elif name.split(".")[0] in {"taiji", "scripts"}:
                missing.append(f"{name} (from {path.relative_to(ROOT).as_posix()})")
    return sorted(seen), missing


def main() -> None:
    started = time.time()
    started_cpu = time.process_time()
    modules, missing = closure(RUNNER)
    rows = []
    for path in modules:
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "role": "runner" if path == RUNNER else "imported module",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for path, role in ASSETS:
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "role": role,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            missing.append(path.relative_to(ROOT).as_posix())
    payload = {
        "format": "taiji-frozen-readset-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window": "taiji training window v2 (2026-11-08), attempt 2",
        "method": "static AST local-import closure; no module imported, no torch, no training",
        "runner": RUNNER.relative_to(ROOT).as_posix(),
        "file_count": len(rows),
        "files": rows,
        "missing": missing,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST)

    lines = [
        "# 禁改清单：在跑训练窗读取集（2026-11-08）",
        "",
        "> 由 `scripts/maintenance/freeze_readset_20261108.py` 静态 AST 解析生成（不导入任何 taiji 模块、不加载权重、不训练）。",
        f"> 生成时间 `{payload['generated_at']}`，共 **{len(rows)}** 个文件；机器可读件 [manifest](../manifests/taiji_frozen_readset_20261108.json)。",
        "",
        "## 约束",
        "",
        "1. 下表任一文件在本次设计工作中**一个字节都不改**；改动只写进新文件（新设计模块/新文档），且新文件不得被 runner 或其 import 闭包引用。",
        "2. 只要下表字节不变，对表外文件的任何编辑都不会影响已在内存中的训练进程，也不会改变续跑的身份判定输入。",
        "3. 续跑身份判定的锚点：预注册 sha（runner 内常量）、runner sha（进程内自算）、检查点逐文件 sha——三者全部落在下表或检查点目录内。",
        "",
        "## 文件与 sha256",
        "",
        "| 角色 | 文件 | 字节 | sha256 |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['role']} | `{row['path']}` | {row['bytes']:,} | `{row['sha256']}` |")
    if missing:
        lines += ["", "## 未解析到的导入（需人工确认是否为本地模块）", ""]
        lines += [f"- `{item}`" for item in missing]
    lines += [
        "",
        "## 已知不一致（登记，不改）",
        "",
        "- runner 文档字符串写「sha fdb975a7」，实际常量与预注册文件均为 `b2db40018e89…`；以常量/文件为准，文档字符串留待下一次允许改 runner 时修正。",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    tmp_doc = DOC.with_suffix(".md.tmp")
    tmp_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_doc.replace(DOC)
    print(
        json.dumps(
            {
                "files": len(rows),
                "missing": missing,
                "manifest": MANIFEST.relative_to(ROOT).as_posix(),
                "doc": DOC.relative_to(ROOT).as_posix(),
                "cpu_seconds": round(time.process_time() - started_cpu, 3),
                "wall_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
