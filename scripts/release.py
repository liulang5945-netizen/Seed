#!/usr/bin/env python3
"""Seed 一键构建脚本 — 统一编排前端构建、PyInstaller 打包、NSIS 安装程序。

用法:
    python scripts/release.py                  # 完整构建（前端 + PyInstaller + NSIS）
    python scripts/release.py --skip-nsis      # 跳过 NSIS（无 NSIS 环境时）
    python scripts/release.py --skip-frontend   # 跳过前端构建（已构建过）
    python scripts/release.py --check-only     # 仅验证产物，不执行构建

输出:
    dist/Seed.exe          — 主程序
    dist/SeedBackend.exe   — 后端进程（Windows）
    dist/SeedSetup.exe     — NSIS 安装程序（如未跳过）
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
FRONTEND_DIST = ROOT / "frontend" / "dist"


def _read_version() -> str:
    """从 pyproject.toml 读取版本号。"""
    import re

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def _run(cmd: list[str], cwd: Path | None = None, label: str = "") -> bool:
    """运行子进程，实时输出。"""
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    print(f"{'=' * 50}")

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=(os.name == "nt"),
    )
    return result.returncode == 0


def _verify_artifacts(skip_nsis: bool) -> list[str]:
    """验证构建产物存在且大小合理。"""
    errors: list[str] = []

    # 前端
    if not (FRONTEND_DIST / "index.html").exists():
        errors.append("frontend/dist/index.html 不存在")

    # PyInstaller 产物
    exe_name = "Seed.exe" if sys.platform == "win32" else "Seed"
    seed_exe = DIST_DIR / exe_name
    if not seed_exe.exists():
        # PyInstaller 可能输出到 dist/Seed/ 子目录
        seed_exe = DIST_DIR / "Seed" / exe_name
        if not seed_exe.exists():
            errors.append(f"{exe_name} 不存在（dist/ 中未找到）")
    else:
        size_mb = seed_exe.stat().st_size / (1024 * 1024)
        if size_mb < 1:
            errors.append(f"{exe_name} 大小异常: {size_mb:.1f} MB（预期 > 1 MB）")

    # NSIS
    if not skip_nsis:
        setup_exe = DIST_DIR / "SeedSetup.exe"
        if not setup_exe.exists():
            errors.append("SeedSetup.exe 不存在")

    return errors


def build_frontend() -> bool:
    """构建前端。"""
    npm = shutil.which("npm") or shutil.which("npm.cmd") or "npm"
    return _run(
        [npm, "run", "build"],
        cwd=ROOT / "frontend",
        label="[1/3] 构建前端",
    )


def build_pyinstaller() -> bool:
    """PyInstaller 打包。"""
    return _run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(ROOT / "desktop" / "seed.spec")],
        cwd=ROOT,
        label="[2/3] PyInstaller 打包",
    )


def build_nsis() -> bool:
    """NSIS 安装程序编译。"""
    makensis = shutil.which("makensis") or shutil.which("makensis.exe")
    if not makensis:
        print("  WARNING: makensis 未找到，跳过 NSIS 编译")
        print("  安装 NSIS: https://nsis.sourceforge.io/")
        return True  # 非致命

    return _run(
        [makensis, str(ROOT / "desktop" / "installer.nsi")],
        cwd=ROOT / "desktop",
        label="[3/3] NSIS 安装程序",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed 一键构建")
    parser.add_argument("--skip-nsis", action="store_true", help="跳过 NSIS 安装程序编译")
    parser.add_argument("--skip-frontend", action="store_true", help="跳过前端构建")
    parser.add_argument("--check-only", action="store_true", help="仅验证产物，不执行构建")
    args = parser.parse_args()

    version = _read_version()
    print(f"Seed v{version} — 构建脚本")

    if args.check_only:
        errors = _verify_artifacts(args.skip_nsis)
        if errors:
            print("\n产物验证失败:")
            for e in errors:
                print(f"  ✗ {e}")
            sys.exit(1)
        print("\n所有产物验证通过")
        return

    # Step 1: Frontend
    if not args.skip_frontend:
        if not build_frontend():
            print("\n前端构建失败")
            sys.exit(1)
        print("  前端构建完成")
    else:
        print("\n  跳过前端构建")

    # Step 2: PyInstaller
    if not build_pyinstaller():
        print("\nPyInstaller 打包失败")
        sys.exit(1)
    print("  PyInstaller 打包完成")

    # Step 3: NSIS
    if not args.skip_nsis:
        if not build_nsis():
            print("\nNSIS 编译失败")
            sys.exit(1)
    else:
        print("\n  跳过 NSIS")

    # Verify
    print("\n验证构建产物...")
    errors = _verify_artifacts(args.skip_nsis)
    if errors:
        print("产物验证失败:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    # Summary
    total = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file())
    print(f"\n{'=' * 50}")
    print(f"  Seed v{version} 构建完成")
    print(f"  输出: {DIST_DIR}")
    print(f"  总大小: {total / 1024 / 1024:.1f} MB")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
