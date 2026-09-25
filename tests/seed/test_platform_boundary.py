"""Product-shell boundary guards for retiring the Legacy Neuroplex runtime."""

from __future__ import annotations

import ast
import os
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]

#: 不是源码面的目录名（依赖树、构建产物、本地虚拟环境、CLI/会话临时 worktree）。
#: 前缀族（`.venv*`、`.dsh-sbx*`）按 R1「规则按命名约定写，不按实例名写」列——
#: 逐个实例名列举必然被下一个同类目录漏掉（`.gitignore:264` 的 `/.dsh-sbx*/` 是同一条约定的锚）。
SKIP_PARTS = {".git", "node_modules", "build", "dist", "_libs", ".codex"}
SKIP_PREFIXES = (".venv", ".dsh-sbx")


def _is_skipped_dir(name: str) -> bool:
    return name in SKIP_PARTS or name.startswith(SKIP_PREFIXES)


def _walk_python_sources(root: Path) -> list[Path]:
    """文件系统遍历（跳过依赖树/构建产物/虚拟环境/会话沙箱）。

    旧守卫写的是 `REPO.rglob("*.py")` 再按 path parts 过滤——**过滤发生在遍历之后**，
    于是 `taiji-harness/node_modules`（pnpm 符号链接森林，仅前四层就 15862 个目录，
    而其中 `.py` 命中数为 **0**）仍被完整遍历；两次全量套件都挂在那里
    （25 秒内 CPU 增量 0.0、无子进程＝阻塞在 reparse-point 的 I/O，不是"慢"）。
    """
    found: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                listed = list(entries)
        except OSError:
            continue
        for entry in listed:
            if entry.is_dir(follow_symlinks=False):
                if _is_skipped_dir(entry.name):
                    continue
                stack.append(Path(entry.path))
            elif entry.name.endswith(".py"):
                found.append(Path(entry.path))
    return found


def _python_sources(root: Path) -> list[Path]:
    """守卫的**扫描面＝版本控制认为存在的那份源码**（tracked ＋ 未被忽略的 untracked）。

    为什么不"修修 rglob 继续扫全仓"：实测 `rglob` 口径下有 30061 个 `.py`，其中
    **28681 个在 `output/`**（训练残留、gitignored）——守卫一直在把非源码当源码扫，
    既制造上面那个挂死，也让"我们的源码没有 BOM"这句话被 28k 个外来文件稀释。
    按 git 取面一次性解决两件事：面就是 CI 检出的面（1718 个），且新增同类沙箱目录
    不需要再往 `_is_skipped_dir` 里补条目（`.gitignore` 才是那条约定的单一出处）。
    `git` 不可用时退回遍历（守卫不许因环境缺工具而静默不判）。
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
            cwd=str(root),
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return _walk_python_sources(root)
    names = [entry for entry in proc.stdout.decode("utf-8").split("\0") if entry]
    found = [root / name for name in names]
    missing = [path for path in found if not path.is_file()]
    if missing:
        raise AssertionError(f"git 列出的源码文件不存在（扫描面失真）：{missing[:5]}")
    return found


def test_source_face_is_the_git_face_not_the_whole_disk() -> None:
    """扫描面必须等于 git 面，且在给定子树里与文件系统遍历**逐集合相等**。

    这条是把"修挂死"与"削弱守卫"分开的凭据：只看 `scanned > 100` 挡不住某一层被悄悄跳过。
    子树里不许有 gitignored 内容，否则两边天然不等——所以断言同时要求
    "遍历比 git 多出来的部分只能是本地噪音（沙箱/依赖/构建目录）"。
    """
    for sub in ("taiji", "api", "scripts/training", "tests/seed"):
        root = REPO / sub
        walked = {p.relative_to(root).as_posix() for p in _walk_python_sources(root)}
        gitted = {p.relative_to(root).as_posix() for p in _python_sources(root)}
        assert walked == gitted, (
            f"{sub}: git 面与遍历面不等（遍历独有 {sorted(walked - gitted)[:5]}，"
            f"git 独有 {sorted(gitted - walked)[:5]}）"
        )
    whole_walked = {p.relative_to(REPO).as_posix() for p in _walk_python_sources(REPO)}
    whole_git = {p.relative_to(REPO).as_posix() for p in _python_sources(REPO)}
    assert whole_git <= whole_walked, "git 面里有遍历看不见的文件（扫描面不可能收窄成这样）"
    extra = sorted(whole_walked - whole_git)
    assert all(
        _is_skipped_dir(PurePosixPath(name).parts[0]) or name.startswith("output/")
        for name in extra
    ), f"多出来的文件不属于任何本地噪音目录，前 5 个：{extra[:5]}"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_api_entrypoint_reaches_legacy_only_through_one_bridge() -> None:
    for entrypoint in ("app.py", "main.py"):
        imports = _imports(REPO / "api" / entrypoint)

        assert "api.legacy_bridge" in imports
        assert not any(module.startswith("neuroplex") for module in imports)


def test_chat_entrypoints_use_the_legacy_gate() -> None:
    for relative in ("api/chat_strategies.py", "api/routes_chat.py"):
        imports = _imports(REPO / relative)
        assert "api.legacy_bridge" in imports, relative
        source = (REPO / relative).read_text(encoding="utf-8")
        assert "legacy_available" in source, relative


def test_desktop_entrypoint_keeps_transformer_dependencies_opt_in() -> None:
    run_app = REPO / "api" / "run_app.py"
    imports = _imports(run_app)
    assert "seed_platform.config" in imports
    assert "seed_platform.dependencies" in imports
    assert "neuroplex.core.config" not in imports

    source = run_app.read_text(encoding="utf-8")
    assert "CORE_DEPENDENCIES" not in source
    assert "transformers" not in source


def test_platform_paths_are_owned_outside_neuroplex() -> None:
    modules = (
        "api/app.py",
        "api/routes_agent_workspace.py",
        "api/routes_chat.py",
        "api/routes_rag.py",
        "api/routes_system.py",
        "api/routes_update.py",
        "api/training/datasets.py",
    )
    for relative in modules:
        imports = _imports(REPO / relative)
        assert "seed_platform.paths" in imports, relative
        assert "neuroplex.core.utils" not in imports, relative

    # publish.py is now a path-free 410 compatibility tombstone; it no longer
    # owns or reads a filesystem location.
    publish_imports = _imports(REPO / "api/training/publish.py")
    assert "seed_platform.paths" not in publish_imports

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"seed_platform*"' in pyproject


def test_frozen_paths_honor_explicit_data_root(monkeypatch) -> None:
    """打包运行可通过显式根目录隔离用户数据，避免写入只读安装位置。"""
    from seed_platform import paths

    explicit_root = REPO / ".seed_test_tmp" / "explicit-data"
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setenv("SEED_DATA_ROOT", str(explicit_root))
    monkeypatch.setattr(paths, "_probe_writable_directory", lambda candidate: True)

    assert paths.get_writable_base_dir() == str(explicit_root)


def test_frozen_paths_fall_back_when_localappdata_is_not_writable(monkeypatch) -> None:
    """LocalAppData 受 ACL 限制时应自动落到包目录的 user_data。"""
    from seed_platform import paths

    test_root = REPO / ".seed_test_tmp" / "paths-fallback"
    local_root = test_root / "localappdata" / "Taiji"
    package_root = test_root / "package"
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "executable", str(package_root / "Seed.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(local_root.parent))
    monkeypatch.delenv("SEED_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        paths,
        "_probe_writable_directory",
        lambda candidate: str(candidate) != str(local_root),
    )

    assert paths.get_writable_base_dir() == str(package_root / "user_data")


def test_platform_state_has_no_legacy_imports() -> None:
    app_state = REPO / "seed_platform" / "app_state.py"
    imports = _imports(app_state)

    assert not any(module.startswith("neuroplex") for module in imports)
    assert "seed_platform.app_state" in _imports(REPO / "neuroplex" / "core" / "app_state.py")

    api_modules = list((REPO / "api").rglob("*.py"))
    for module_path in api_modules:
        imports = _imports(module_path)
        assert "neuroplex.core.app_state" not in imports, module_path


def test_runtime_status_keeps_legacy_sections_opt_in() -> None:
    source = (REPO / "seed_platform" / "runtime_service.py").read_text(encoding="utf-8")
    assert "legacy_requested" in source
    assert "neuroplex.life.life_scheduler" in source
    assert "from seed_platform.workbench import CapabilitySnapshot" in source
    assert "neuroplex.services.tool_service" not in source


def test_platform_auth_has_no_legacy_imports() -> None:
    auth = REPO / "seed_platform" / "auth.py"
    imports = _imports(auth)

    assert not any(module.startswith("neuroplex") for module in imports)
    assert "seed_platform.auth" in _imports(REPO / "neuroplex" / "core" / "security.py")

    api_modules = list((REPO / "api").rglob("*.py"))
    for module_path in api_modules:
        imports = _imports(module_path)
        assert "neuroplex.core.security" not in imports, module_path
        assert "neuroplex.services.auth_service" not in imports, module_path


def test_legacy_bridge_owns_explicit_cortex_routes_and_lifecycle() -> None:
    bridge = REPO / "api" / "legacy_bridge.py"
    text = bridge.read_text(encoding="utf-8")

    assert "register_legacy_routers" in text
    assert "load_legacy_runtime" in text
    assert "start_legacy_services" in text
    assert "stop_legacy_services" in text
    assert "legacy_available" in text


def test_python_sources_have_no_utf8_bom() -> None:
    # BOM 是隐形炸弹：black 走 tokenize.open 会静默剥离，CI 因此长绿，
    # 但任何 ast.parse(read_text(encoding="utf-8")) 都会炸 U+FEFF。
    # scripts/archive/ 内的脚本已因历史 mojibake 无法解析，不在守卫范围。
    # 依赖树/构建产物/本地虚拟环境由 `_python_sources` 在**遍历时**就排除（见其 docstring）。
    scanned = 0
    offenders: list[str] = []
    for path in _python_sources(REPO):
        relative = path.relative_to(REPO).as_posix()
        if relative.startswith("scripts/archive/"):
            continue
        scanned += 1
        if path.read_bytes().startswith(b"\xef\xbb\xbf"):
            offenders.append(relative)

    assert scanned > 100, f"BOM 扫描面异常收窄，仅扫到 {scanned} 个文件"
    assert offenders == [], f"以下 Python 源码带 UTF-8 BOM：{offenders}"
