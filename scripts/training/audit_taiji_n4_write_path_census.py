"""N4「写入真发生在产品链」的取数普查（零算力，只读源码；㊵-601 留下的 J-N4d-3 前置）。

**为什么要有这台仪器**：05/08 里"情景记忆写入"这句话长期只有两种说法——"产品默认位是关的所以没写"
与"训练器每篇文档都在写"。两句都对，但**分属不同的链**，而 J-N4d-3（`mount_layer="product"` 的寻址面读数）
要的恰恰是"哪一层在写、默认下发不发生、寻址读数来自哪一层"。散文答不了，枚举能答。

**取法与限度（随件出版，不许被读成运行时证明）**：本器只回答三件事——
①代码里有没有对情景库的 `.write(` 调用点及其**守卫条件**文本；②`attach_episodic_memory` 的挂载点有几处、
分别在哪；③产品默认位读自 `taiji/config.py` 的字段默认值（现读，不抄常量）。
它**不**证明某次运行里真的走到了：那是在场计数器（另一格）的活。`reading_limit` 字段把这条写进件里。

**rc 语义**：`0`＝普查跑完（无论结论是什么）；`2`＝取法失败（目标文件不在、config 里读不到那两枚字段）
⇒ 那是"没算"，不许被读成"没有写入点"。
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = "taiji/config.py"
TRAINER_FILE = "scripts/training/train_seed_corpus.py"
FACES = ("taiji", "seed", "api", "scripts/training")

#: 名字里含这个子串的对象上的 `.write(` 才算情景库写入点。
EPISODIC_TOKEN = "episodic"
MOUNT_METHOD = "attach_episodic_memory"
#: 守卫文本＝调用点往上最多 6 行里最近的一条以 `if ` 开头的源码行（拿不到就出版 None，不猜）。
DEFAULT_FIELDS = ("episodic_memory_default_mount", "episodic_memory_capacity")


def _config_defaults(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id in names and node.value is not None:
            found[node.target.id] = ast.literal_eval(node.value)
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(f"config lacks {missing}")
    return found


def _guard_text(lines: list[str], index: int) -> str | None:
    for offset in range(index - 1, max(index - 7, -1), -1):
        stripped = lines[offset].strip()
        if stripped.startswith("if "):
            return stripped
    return None


def _identifier_of(node: ast.AST) -> str:
    """取调用接受者的名字：`a.write(...)`→`a`，`a.b.write(...)`→`a.b`。"""

    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _scan(root: Path) -> dict[str, Any]:
    """AST 扫描调用点。

    〔2026-10-09 ㊵-643 第一版的两处取法错，各朝一个方向〕：我原来用一条正则去匹配
    `<标识符含 episodic>.write(`——它**漏报**了以 `episodic_` 开头的名字（`episodic_store.write(...)`
    在训练器里真实存在，读数却是 0），又**误报**了 docstring 里提到的同名串
    （`make_taiji_n4_product_tier_faces.py` 的第 5 行是说明文字）。正则的字符类要求
    "episodic" 前面至少有一个字符，所以首字母就是 e 的名字全部漏掉。改成 AST 之后这两侧都能为假，
    并由契约测钉住（漏报支与误报支各一支）。
    """

    writes: list[dict[str, Any]] = []
    mounts: list[dict[str, Any]] = []
    seen_files = 0
    for face in FACES:
        directory = root / face
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            seen_files += 1
            source = path.read_text(encoding="utf-8", errors="replace")
            lines = source.split(chr(10))
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            relative = path.relative_to(root).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                receiver = _identifier_of(node.func.value)
                if node.func.attr == "write" and EPISODIC_TOKEN in receiver.lower():
                    writes.append(
                        {
                            "file": relative,
                            "line": int(node.lineno),
                            "target": receiver,
                            "guarded_by": _guard_text(lines, node.lineno - 1),
                        }
                    )
                if node.func.attr == MOUNT_METHOD:
                    mounts.append({"file": relative, "line": int(node.lineno)})
    return {"write_sites": writes, "mount_sites": mounts, "files_scanned": seen_files}


def audit(root: Path) -> dict[str, Any]:
    config_path = root / CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    defaults = _config_defaults(config_path, DEFAULT_FIELDS)
    scan = _scan(root)
    writes = scan["write_sites"]
    trainer_writes = [w for w in writes if w["file"] == TRAINER_FILE]
    product_writes = [w for w in writes if w["file"].startswith(("taiji/", "seed/", "api/"))]
    mount_total = len(scan["mount_sites"])
    adapter_mounts = [m for m in scan["mount_sites"] if m["file"] == "taiji/adapter.py"]
    return {
        "format": "taiji-n4-write-path-census-v1",
        "defaults": defaults,
        "write_sites": writes,
        "write_site_count": len(writes),
        "trainer_write_site_count": len(trainer_writes),
        "product_write_site_count": len(product_writes),
        "mount_sites": scan["mount_sites"],
        "mount_site_count": mount_total,
        "adapter_mount_site_count": len(adapter_mounts),
        #: 三条判定各自独立，且都不越过本器的取法限度。
        "verdicts": {
            "product_chain_has_write_call": bool(product_writes),
            "trainer_chain_has_write_call": bool(trainer_writes),
            "write_happens_at_product_default": bool(
                defaults["episodic_memory_default_mount"] and product_writes
            ),
        },
        "reading_limit": (
            "call-site presence and the config default only; runtime reachability needs an "
            "on-face presence counter (not measured here)"
        ),
        "files_scanned": scan["files_scanned"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N4 write-path census (static, zero-compute)")
    parser.add_argument("--out-report", type=Path, required=True, help="where to write the JSON")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="tree to scan")
    args = parser.parse_args(argv)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = audit(args.root)
    except (FileNotFoundError, ValueError, SyntaxError) as error:
        args.out_report.write_text(
            json.dumps(
                {
                    "format": "taiji-n4-write-path-census-v1",
                    "status": "census_failed",
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return 2
    args.out_report.write_text(
        json.dumps({"status": "ok", **payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
