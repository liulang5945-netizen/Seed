# -*- mode: python ; coding: utf-8 -*-
"""Seed 桌面端三入口打包规格（由 scripts/release.py 调用）。

- Seed.exe        : GUI 主入口（windowed，desktop/main.py）
- SeedBackend.exe : 后端工作进程（console，desktop/backend_worker.py）
- SeedWs.exe      : 8765 WebSocket 服务器工作进程（console，desktop/seed_ws.py）

三个入口经 MERGE 共享同一份 _internal 依赖，避免体积翻倍。
frozen 模式下主程序以子进程拉起 SeedBackend.exe，等价于开发模式
的 `python -m uvicorn api.app:app`，规避：
1. `sys.executable -m uvicorn` 递归启动 GUI 的问题；
2. 进程内线程/多进程方案与 logging 配置、PyInstaller spawn 的冲突。

SeedWs.exe 目前只被 Electron 壳（desktop-electron/）使用：Node 进程无法 import
Python 模块，8765 必须有独立子进程入口。PyQt6 侧的 frozen 分支用进程内守护线程跑
同一模块（见 desktop/main.py: WebSocketManager._start_inproc），因此**新增该入口
对 PyQt6 出货路径无任何行为影响**。
"""
import importlib.util
import os
from pathlib import Path

ROOT = Path(os.environ.get("SEED_BUILD_ROOT", Path(SPECPATH).parent))

block_cipher = None

_common_hiddenimports = [
    "neuroplex", "api", "uvicorn", "fastapi", "pydantic",
    "torch",
]

# Semantic providers are opt-in and may not be installed on a native-only
# build host.  Always freeze the Seed adapter contract; include the optional
# Qwen dependency graph only when the build environment actually provides it.
# This keeps the packaged client safe without silently baking a model path or
# making Transformers a mandatory product dependency.
_semantic_provider_hiddenimports = ["seed", "seed.semantic_provider"]
for _optional_module in ("transformers", "safetensors", "sentencepiece", "accelerate"):
    if importlib.util.find_spec(_optional_module) is not None:
        _semantic_provider_hiddenimports.append(_optional_module)

a_backend = Analysis(
    [str(ROOT / "desktop" / "backend_worker.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=_common_hiddenimports + _semantic_provider_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_ws = Analysis(
    [str(ROOT / "desktop" / "seed_ws.py")],
    pathex=[str(ROOT)],
    binaries=[],
    # 数据资产不在此声明：MERGE 让三个 exe 共享同一份 _internal，且 COLLECT 只列
    # a_main.datas，因此 a_ws 运行时同样能看到全部数据，此处重复声明只会造成重复条目。
    datas=[],
    hiddenimports=_common_hiddenimports + _semantic_provider_hiddenimports + ["websockets"],
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


def _append_data_tree(source: Path, destination: str) -> None:
    """把目录展开为显式文件项，避免 PyInstaller 复用旧 Tree 条目。"""

    if not source.is_dir():
        return
    destination_root = Path(destination)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative_parent = path.relative_to(source).parent
            _datas.append((str(path), str(destination_root / relative_parent)))


# 前端必须逐文件枚举。直接把 frontend/dist 作为一个目录交给 PyInstaller
# 时，增量分析可能保留上一轮带 hash 文件名的 TOC；Vite 已换 hash 后，
# COLLECT 会静默跳过新的入口 JS/CSS，最终只剩 index.html 字节校验能通过，
# 客户端却在真实窗口中空白。
_append_data_tree(ROOT / "frontend" / "dist", "frontend/dist")

for src, dst in [
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
    hiddenimports=_common_hiddenimports + _semantic_provider_hiddenimports + [
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

# Qt6Core on Windows intentionally resolves ``icuuc.dll`` from the operating
# system.  A transitive ML dependency can otherwise contribute ICU 78 under
# the same basename at the bundle root; that DLL is not ABI-compatible with
# the Windows ICU used by the PyQt6 wheel and makes QtCore.pyd fail with
# WinError 127 before the desktop window is created.
_system_icu_basenames = {"icuuc.dll", "icudt78.dll"}
for _analysis in (a_main, a_backend, a_ws):
    _analysis.binaries = TOC(
        entry
        for entry in _analysis.binaries
        if Path(entry[0]).name.lower() not in _system_icu_basenames
    )

# 合并重复模块，三个 exe 共享 _internal（新版 PyInstaller 需三元组：
# (analysis, identifier, path_to_exe)）
MERGE(
    (a_main, "seed-main", "Seed"),
    (a_backend, "seed-backend", "SeedBackend"),
    (a_ws, "seed-ws", "SeedWs"),
)

pyz_main = PYZ(a_main.pure, a_main.zipped_data, cipher=block_cipher)
pyz_backend = PYZ(a_backend.pure, a_backend.zipped_data, cipher=block_cipher)
pyz_ws = PYZ(a_ws.pure, a_ws.zipped_data, cipher=block_cipher)

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

exe_ws = EXE(
    pyz_ws,
    a_ws.scripts,
    [],
    exclude_binaries=True,
    name="SeedWs",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe_main,
    exe_backend,
    exe_ws,
    a_main.binaries,
    a_main.zipfiles,
    a_main.datas,
    a_ws.binaries,
    a_ws.zipfiles,
    a_ws.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Seed",
)
