# -*- mode: python ; coding: utf-8 -*-
"""Seed 桌面端双入口打包规格（由 scripts/release.py 调用）。

- Seed.exe        : GUI 主入口（windowed，desktop/main.py）
- SeedBackend.exe : 后端工作进程（console，desktop/backend_worker.py）

两个入口经 MERGE 共享同一份 _internal 依赖，避免体积翻倍。
frozen 模式下主程序以子进程拉起 SeedBackend.exe，等价于开发模式
的 `python -m uvicorn api.app:app`，规避：
1. `sys.executable -m uvicorn` 递归启动 GUI 的问题；
2. 进程内线程/多进程方案与 logging 配置、PyInstaller spawn 的冲突。
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SEED_BUILD_ROOT", Path(SPECPATH).parent))

block_cipher = None

_common_hiddenimports = [
    "neuroplex", "api", "uvicorn", "fastapi", "pydantic",
    "torch",
]

a_backend = Analysis(
    [str(ROOT / "desktop" / "backend_worker.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=_common_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 运行时数据资产（PyInstaller 只收集 .py，纯数据文件必须显式声明）：
# - tokenizer_contract.json / domains/*.model：cortex 装配与 life 状态必需，
#   缺失会让 /api/health 直接 500；
# - checkpoints/seed_corpus.pt：Seed 原生运行时激活用。
_datas = []
for src, dst in [
    (ROOT / "frontend" / "dist", "frontend/dist"),
    (ROOT / "neuroplex" / "tokenizer_contract.json", "neuroplex"),
    (ROOT / "neuroplex" / "domains", "neuroplex/domains"),
    (ROOT / "checkpoints" / "seed_corpus.pt", "checkpoints"),
    (ROOT / "taiji_data" / "final", "taiji_data/final"),
    (ROOT / "app_settings.json", "."),
    (ROOT / "version.json", "."),
    (ROOT / "icon.ico", "."),
]:
    if src.exists():
        _datas.append((str(src), dst))

a_main = Analysis(
    [str(ROOT / "desktop" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=_datas,
    hiddenimports=_common_hiddenimports + [
        "PyQt6", "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebChannel",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 合并重复模块，两个 exe 共享 _internal（新版 PyInstaller 需三元组：
# (analysis, identifier, path_to_exe)）
MERGE((a_main, "seed-main", "Seed"), (a_backend, "seed-backend", "SeedBackend"))

pyz_main = PYZ(a_main.pure, a_main.zipped_data, cipher=block_cipher)
pyz_backend = PYZ(a_backend.pure, a_backend.zipped_data, cipher=block_cipher)

_icon = str(ROOT / "icon.ico") if (ROOT / "icon.ico").exists() else str(ROOT / "frontend" / "public" / "favicon.ico")

exe_main = EXE(
    pyz_main,
    a_main.scripts,
    [],
    exclude_binaries=True,
    name="Seed",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=_icon,
)

exe_backend = EXE(
    pyz_backend,
    a_backend.scripts,
    [],
    exclude_binaries=True,
    name="SeedBackend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe_main,
    exe_backend,
    a_main.binaries,
    a_main.zipfiles,
    a_main.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Seed",
)
