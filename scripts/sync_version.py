#!/usr/bin/env python3
"""同步版本号：以 pyproject.toml 为唯一事实来源，更新其余文件。

用法:
    python scripts/sync_version.py            # 从 pyproject.toml 读取版本并同步
    python scripts/sync_version.py 1.7.0      # 先设置新版本号再同步
    python scripts/sync_version.py --check    # 仅检查是否一致（CI 用）
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

# (relative_path, pattern_or_replacer)
TARGETS = [
    # frontend/package.json  →  "version": "X.Y.Z"
    (
        "frontend/package.json",
        lambda content, ver: re.sub(
            r'("version"\s*:\s*)"[^"]*"', rf'\g<1>"{ver}"', content, count=1
        ),
    ),
    # desktop/installer.nsi  →  !define APP_VERSION "X.Y.Z"
    # 注意：替换串必须用单引号 f-string 写出闭合双引号，raw string 里的 \" 会把反斜杠写进文件
    (
        "desktop/installer.nsi",
        lambda content, ver: re.sub(
            r'(!define\s+APP_VERSION\s+")[^"]*"', rf'\g<1>{ver}"', content, count=1
        ),
    ),
    # desktop/main.py  →  setApplicationVersion("X.Y.Z")  +  SeedDesktop/X.Y.Z
    (
        "desktop/main.py",
        lambda content, ver: (
            re.sub(
                r'setApplicationVersion\("[^"]*"\)',
                f'setApplicationVersion("{ver}")',
                content,
            ).replace(
                (
                    re.search(r'SeedDesktop/[^"]*', content).group(0)
                    if re.search(r'SeedDesktop/[^"]*', content)
                    else ""
                ),
                f"SeedDesktop/{ver}",
            )
            if re.search(r"SeedDesktop/[^\"]*", content)
            else re.sub(
                r'setApplicationVersion\("[^"]*"\)',
                f'setApplicationVersion("{ver}")',
                content,
            )
        ),
    ),
    # desktop/loading.html  →  Seed vX.Y.Z
    (
        "desktop/loading.html",
        lambda content, ver: re.sub(
            r"Seed v[0-9]+\.[0-9]+\.[0-9]+", f"Seed v{ver}", content, count=1
        ),
    ),
]


def read_version() -> str:
    """从 pyproject.toml 读取版本号。"""
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise SystemExit("ERROR: 无法在 pyproject.toml 中找到 version 字段")
    return m.group(1)


def set_version(ver: str) -> None:
    """写入 pyproject.toml 版本号。"""
    text = PYPROJECT.read_text(encoding="utf-8")
    text = re.sub(
        r'(^version\s*=\s*)"[^"]*"',
        rf'\g<1>"{ver}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    PYPROJECT.write_text(text, encoding="utf-8")


def sync(dry_run: bool = False) -> list[str]:
    """同步版本号到所有目标文件。返回不一致列表（dry_run 模式）。"""
    ver = read_version()
    drifts: list[str] = []

    for rel_path, replacer in TARGETS:
        fp = ROOT / rel_path
        if not fp.exists():
            print(f"  SKIP {rel_path} (not found)")
            continue
        original = fp.read_text(encoding="utf-8")
        updated = replacer(original, ver)
        if original == updated:
            print(f"  OK   {rel_path}")
        else:
            drifts.append(rel_path)
            if dry_run:
                print(f"  DRIFT {rel_path}")
            else:
                fp.write_text(updated, encoding="utf-8")
                print(f"  FIX  {rel_path}")

    return drifts


def main() -> None:
    args = sys.argv[1:]
    check_mode = "--check" in args
    args = [a for a in args if a != "--check"]

    if args:
        new_ver = args[0]
        print(f"设置版本号: {new_ver}")
        set_version(new_ver)

    if check_mode:
        print("检查版本一致性...")
        drifts = sync(dry_run=True)
        if drifts:
            print(f"\n{len(drifts)} 个文件版本不一致: {', '.join(drifts)}")
            print("运行 `python scripts/sync_version.py` 修复")
            sys.exit(1)
        else:
            print("所有文件版本一致 (OK)")
    else:
        print("同步版本号...")
        drifts = sync()
        if drifts:
            print(f"\n已更新 {len(drifts)} 个文件")
        else:
            print("\n所有文件已是最新")


if __name__ == "__main__":
    main()
