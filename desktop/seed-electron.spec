# -*- mode: python ; coding: utf-8 -*-
"""Seed Electron 专用打包规格（由 scripts/release.py --electron 调用）。

- SeedBackend.exe : 后端工作进程（console，desktop/backend_worker.py）
- SeedWs.exe      : 8765 WebSocket 服务器工作进程（console，desktop/seed_ws.py）

**没有 Seed.exe**：Electron 壳自己就是 GUI（desktop-electron/），PyQt6 的 GUI 入口
在这里只会带来约 200 MB 的 Qt6 运行时冗余（含 QtWebEngineProcess.exe）。
PyQt6 出货路径请继续用 desktop/seed.spec（三入口，含 Seed.exe）。

与 seed.spec 的差异清单（逐项）：
1. 去掉 a_main（PyQt6 GUI）及其 hiddenimports 里的 PyQt6.*;
2. 数据资产改挂在 a_backend 上（原来挂在 a_main，经 COLLECT 汇入 _internal）；
3. MERGE 只有两项；COLLECT 不再有 exe_main；
4. COLLECT name 仍为 "Seed"，使 desktop-electron/electron-builder.yml 的
   extraFiles（from: ../dist/Seed）无需改动。

frozen 模式下 Electron 主进程以子进程拉起 SeedBackend.exe，等价于开发模式的
`python -m uvicorn api.app:app`；SeedWs.exe 等价于 `python -m neuroplex.core.websocket_server`。
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

# 运行时数据资产（PyInstaller 只收集 .py，纯数据文件必须显式声明）：
# - tokenizer_contract.json / domains/*.model：cortex 装配与 life 状态必需，
#   缺失会让 /api/health 直接 500；
# - checkpoints/seed_corpus.pt：Seed 原生运行时激活用；
# - frontend/dist：由 Python 后端自己托管（api/app.py::_mount_static_assets + SPA catch-all），
#   Electron 壳不打包前端，这点与 seed.spec 一致。
#
# 前端必须逐文件枚举。直接把 frontend/dist 作为一个目录交给 PyInstaller 时，
# 增量分析可能保留上一轮带 hash 文件名的 TOC；Vite 已换 hash 后，COLLECT 会静默跳过
# 新的入口 JS/CSS，最终只剩 index.html 字节校验能通过，客户端却在真实窗口中空白。
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

a_backend = Analysis(
    [str(ROOT / "desktop" / "backend_worker.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=_datas,
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
    # 数据资产不在此声明：MERGE 让两个 exe 共享同一份 _internal，且 COLLECT 只列
    # a_backend.datas，因此 a_ws 运行时同样能看到全部数据，此处重复声明只会造成重复条目。
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

# Qt6Core on Windows intentionally resolves ``icuuc.dll`` from the operating
# system.  A transitive ML dependency can otherwise contribute ICU 78 under
# the same basename at the bundle root; that DLL is not ABI-compatible with
# the Windows ICU used by the PyQt6 wheel and makes QtCore.pyd fail with
# WinError 127 before the desktop window is created.
# Electron 轨道虽然不带 PyQt6，但传递依赖仍可能拖进 ICU，故过滤保留。
_system_icu_basenames = {"icuuc.dll", "icudt78.dll"}
for _analysis in (a_backend, a_ws):
    _analysis.binaries = TOC(
        entry
        for entry in _analysis.binaries
        if Path(entry[0]).name.lower() not in _system_icu_basenames
    )

# 合并重复模块，两个 exe 共享 _internal（新版 PyInstaller 需三元组：
# (analysis, identifier, path_to_exe)）
MERGE(
    (a_backend, "seed-backend", "SeedBackend"),
    (a_ws, "seed-ws", "SeedWs"),
)

pyz_backend = PYZ(a_backend.pure, a_backend.zipped_data, cipher=block_cipher)
pyz_ws = PYZ(a_ws.pure, a_ws.zipped_data, cipher=block_cipher)

_icon = str(ROOT / "icon.ico") if (ROOT / "icon.ico").exists() else str(ROOT / "frontend" / "public" / "favicon.ico")

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
    exe_backend,
    exe_ws,
    a_backend.binaries,
    a_backend.zipfiles,
    a_backend.datas,
    a_ws.binaries,
    a_ws.zipfiles,
    a_ws.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Seed",
)
