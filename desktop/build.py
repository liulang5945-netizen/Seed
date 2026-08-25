"""
Seed桌面客户端打包脚本
========================

使用 PyInstaller 打包为独立可执行文件。

使用方式：
    python desktop/build.py

输出：
    dist/Seed.exe (Windows)
    dist/Seed (Linux/Mac)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"


def _verify_packaged_frontend() -> None:
    """保证冻结客户端携带的前端就是本次构建出来的前端。

    开发目录的 ``frontend/dist`` 与 PyInstaller 的
    ``dist/Seed/_internal/frontend/dist`` 是两份独立副本。只启动旧 exe
    时，源码中的视觉改动不会生效；用 index.html 做字节级断言可以把这种
    静默的版本漂移变成明确的构建失败。
    """
    source_index = ROOT_DIR / "frontend" / "dist" / "index.html"
    packaged_index = DIST_DIR / "Seed" / "_internal" / "frontend" / "dist" / "index.html"
    if not source_index.is_file():
        raise RuntimeError(f"前端构建产物不存在: {source_index}")
    if not packaged_index.is_file():
        raise RuntimeError(f"打包产物未包含前端: {packaged_index}")

    source_bytes = source_index.read_bytes()
    packaged_bytes = packaged_index.read_bytes()
    if source_bytes != packaged_bytes:
        raise RuntimeError(
            "打包客户端内置前端与本次构建产物不一致；"
            "请检查 PyInstaller datas 是否仍指向 frontend/dist"
        )

    print("  前端一致性校验通过（源码 dist = 客户端内置 dist）")


def build():
    """打包Seed桌面客户端"""
    print("=" * 50)
    print("  Seed桌面客户端打包")
    print("=" * 50)

    # 清理旧构建
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    # 前端构建（Windows 上 npm 实为 npm.cmd，直接调 "npm" 会 WinError 2）
    print("\n[1/3] 构建前端...")
    frontend_dir = ROOT_DIR / "frontend"
    npm_cmd = shutil.which("npm") or shutil.which("npm.cmd") or "npm"
    result = subprocess.run(
        [npm_cmd, "run", "build"],
        cwd=str(frontend_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # npm 中文输出编码不稳定，避免解码崩溃；内容仅用于报错展示
        shell=(os.name == "nt"),  # Windows 下 .cmd 需要 shell 解析
    )
    if result.returncode != 0:
        print(f"前端构建失败: {result.stderr}")
        return False
    print("  前端构建完成")

    # PyInstaller 打包（双入口：Seed.exe GUI + SeedBackend.exe 后端，
    # 见 desktop/seed.spec；MERGE 共享 _internal 避免体积翻倍）
    print("\n[2/3] PyInstaller 打包...")
    print("  规格: desktop/seed.spec")

    # 应用图标：优先根目录 icon.ico，回退前端 favicon.ico（spec 内同样逻辑）
    icon_file = ROOT_DIR / "icon.ico"
    if not icon_file.exists():
        icon_file = ROOT_DIR / "frontend" / "public" / "favicon.ico"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        str(ROOT_DIR / "desktop" / "seed.spec"),
    ]

    print(f"  命令: {' '.join(cmd[:5])}...")
    result = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # PyInstaller 输出含中文路径/警告时同样容错
    )

    if result.returncode != 0:
        print(f"打包失败:\n{result.stderr[-500:]}")
        return False

    print("  PyInstaller 打包完成")
    _verify_packaged_frontend()

    # 后处理
    print("\n[3/3] 后处理...")

    # 复制额外文件到 dist
    dist_seed = DIST_DIR / "Seed"
    if dist_seed.exists():
        # 复制知识库目录
        for extra_dir in ["knowledge_store", "user_data", "security"]:
            src = ROOT_DIR / extra_dir
            if src.exists():
                shutil.copytree(src, dist_seed / extra_dir, dirs_exist_ok=True)

        # 创建空目录
        for empty_dir in [
            "agent_workspace",
            "taiji_data/feed_data",
            "taiji_data/sleep_data",
            "taiji_data/life_data",
            "taiji_data/evolution_data",
        ]:
            (dist_seed / empty_dir).mkdir(parents=True, exist_ok=True)

    print("  后处理完成")

    # 统计
    print("\n" + "=" * 50)
    total_size = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file())
    print(f"  输出目录: {DIST_DIR}")
    print(f"  总大小: {total_size / 1024 / 1024:.1f} MB")
    print("=" * 50)

    return True


if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
