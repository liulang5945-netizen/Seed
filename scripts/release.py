#!/usr/bin/env python3
"""Seed 一键构建脚本 — 统一编排前端构建、PyInstaller 打包、后处理、NSIS 安装程序。

这是**唯一**的桌面发布入口（原 desktop/build.py 已并入此处并删除）。

用法:
    python scripts/release.py                  # 完整构建（前端 + PyInstaller + 后处理 + NSIS）
    python scripts/release.py --skip-nsis      # 跳过 NSIS（无 NSIS 环境时）
    python scripts/release.py --skip-frontend   # 跳过前端构建（已构建过）
    python scripts/release.py --no-clean       # 保留旧 dist/build（增量调试用）
    python scripts/release.py --check-only     # 仅验证产物，不执行构建

输出:
    dist/Seed/Seed.exe          — 主程序（GUI）
    dist/Seed/SeedBackend.exe   — 后端进程（Windows）
    dist/SeedSetup.exe          — NSIS 安装程序（如未跳过）
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
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


def _verify_packaged_frontend() -> None:
    """保证冻结客户端携带的前端就是本次构建出来的前端。

    开发目录的 ``frontend/dist`` 与 PyInstaller 的
    ``dist/Seed/_internal/frontend/dist`` 是两份独立副本。只启动旧 exe 时，
    源码中的视觉改动不会生效；用 index.html 做字节级断言可以把这种静默的
    版本漂移变成明确的构建失败。
    """
    source_index = FRONTEND_DIST / "index.html"
    packaged_index = DIST_DIR / "Seed" / "_internal" / "frontend" / "dist" / "index.html"
    if not source_index.is_file():
        raise RuntimeError(f"前端构建产物不存在: {source_index}")
    if not packaged_index.is_file():
        raise RuntimeError(f"打包产物未包含前端: {packaged_index}")
    if source_index.read_bytes() != packaged_index.read_bytes():
        raise RuntimeError(
            "打包客户端内置前端与本次构建产物不一致；"
            "请检查 PyInstaller datas 是否仍指向 frontend/dist"
        )
    print("  前端一致性校验通过（源码 dist = 客户端内置 dist）")


def _verify_artifacts(skip_nsis: bool) -> list[str]:
    """验证构建产物存在且大小合理。"""
    errors: list[str] = []

    # 前端
    if not (FRONTEND_DIST / "index.html").exists():
        errors.append("frontend/dist/index.html 不存在")

    # PyInstaller 产物：seed.spec 的 COLLECT name 为 "Seed"，故落在 dist/Seed/ 下
    is_win = sys.platform == "win32"
    for exe_stem in ("Seed", "SeedBackend"):
        if exe_stem == "SeedBackend" and not is_win:
            continue  # SeedBackend 仅 Windows 双进程方案需要
        exe_name = f"{exe_stem}.exe" if is_win else exe_stem
        exe_path = DIST_DIR / "Seed" / exe_name
        if not exe_path.exists():
            errors.append(f"{exe_name} 不存在（dist/Seed/ 中未找到）")
            continue
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        if size_mb < 1:
            errors.append(f"{exe_name} 大小异常: {size_mb:.1f} MB（预期 > 1 MB）")

    # 内置前端一致性：把「改了前端却打出旧包」变成显式失败
    try:
        _verify_packaged_frontend()
    except RuntimeError as exc:
        errors.append(str(exc))

    # NSIS（installer.nsi 的 OutFile 为 ..\dist\SeedSetup.exe）
    if not skip_nsis:
        if not (DIST_DIR / "SeedSetup.exe").exists():
            errors.append("dist/SeedSetup.exe 不存在")

    return errors


def clean_outputs() -> None:
    """清理旧的 dist/build，避免上一轮残留产物混入本次发布。"""
    for target in (DIST_DIR, BUILD_DIR):
        if target.exists():
            shutil.rmtree(target)
            print(f"  已清理 {target.relative_to(ROOT)}")


def postprocess() -> None:
    """后处理：补齐冻结产物运行所需的可写目录与随包数据。"""
    dist_seed = DIST_DIR / "Seed"
    if not dist_seed.exists():
        raise RuntimeError(f"打包目录不存在: {dist_seed}")

    for extra_dir in ("knowledge_store", "user_data", "security"):
        src = ROOT / extra_dir
        if src.exists():
            shutil.copytree(src, dist_seed / extra_dir, dirs_exist_ok=True)
            print(f"  已随包复制 {extra_dir}/")

    for empty_dir in (
        "agent_workspace",
        "taiji_data/feed_data",
        "taiji_data/sleep_data",
        "taiji_data/life_data",
        "taiji_data/evolution_data",
    ):
        (dist_seed / empty_dir).mkdir(parents=True, exist_ok=True)
    print("  运行时可写目录已就绪")


def _npm() -> str:
    """Windows 上 npm 实为 npm.cmd，直接调 "npm" 会 WinError 2。"""
    return shutil.which("npm") or shutil.which("npm.cmd") or "npm"


def check_generated_sources() -> bool:
    """校验「由依赖反推、写死进源码」的构建期产物仍与依赖同步。

    当前唯一一项是 ``frontend/src/composables/hljsAliases.js``：它的 179 条
    别名是从 highlight.js 反推出来的。升级 highlight.js 后该表不会报错、
    不会警告，只会让新语言的代码块静默退化成无高亮纯文本——而全部单元测试
    依然全绿。这类漂移必须在产出安装包之前硬失败，否则门禁虽存在却不在
    发版的必经路径上。

    该检查不受 ``--skip-frontend`` 影响：跳过的是构建，不是校验，且既有
    ``frontend/dist`` 正是由这份可能已过期的源码产出的。
    """
    return _run(
        [_npm(), "run", "check:aliases"],
        cwd=ROOT / "frontend",
        label="[1/5] 生成式源码同步门禁",
    )


def build_frontend() -> bool:
    """构建前端。"""
    return _run(
        [_npm(), "run", "build"],
        cwd=ROOT / "frontend",
        label="[2/5] 构建前端",
    )


def build_pyinstaller() -> bool:
    """PyInstaller 打包（双入口 Seed.exe + SeedBackend.exe，见 desktop/seed.spec）。"""
    return _run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(ROOT / "desktop" / "seed.spec")],
        cwd=ROOT,
        label="[3/5] PyInstaller 打包",
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
        label="[5/5] NSIS 安装程序",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed 一键构建")
    parser.add_argument("--skip-nsis", action="store_true", help="跳过 NSIS 安装程序编译")
    parser.add_argument("--skip-frontend", action="store_true", help="跳过前端构建")
    parser.add_argument("--no-clean", action="store_true", help="保留旧 dist/build")
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

    # Step 1: 生成式源码同步门禁（不受 --skip-frontend 影响，理由见函数 docstring）
    # 置于清理之前：脱同步是必然中止的错误，不该先把上一版可用产物删掉再报错。
    if not check_generated_sources():
        print("\n生成式源码已与依赖脱同步，构建中止")
        print("  修复: cd frontend && npm run gen:aliases")
        sys.exit(1)
    print("  生成式源码检查通过")

    # Step 0: 清理旧产物（默认执行；否则 PyInstaller 的增量复用会掩盖版本漂移）
    if not args.no_clean:
        print("\n清理旧产物...")
        clean_outputs()

    # Step 2: Frontend
    if not args.skip_frontend:
        if not build_frontend():
            print("\n前端构建失败")
            sys.exit(1)
        print("  前端构建完成")
    else:
        print("\n  跳过前端构建")

    # Step 3: PyInstaller
    if not build_pyinstaller():
        print("\nPyInstaller 打包失败")
        sys.exit(1)
    print("  PyInstaller 打包完成")

    try:
        _verify_packaged_frontend()
    except RuntimeError as exc:
        print(f"\n前端一致性校验失败: {exc}")
        sys.exit(1)

    # Step 4: 后处理
    print(f"\n{'=' * 50}")
    print("  [4/5] 后处理")
    print(f"{'=' * 50}")
    try:
        postprocess()
    except RuntimeError as exc:
        print(f"\n后处理失败: {exc}")
        sys.exit(1)

    # Step 5: NSIS
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
