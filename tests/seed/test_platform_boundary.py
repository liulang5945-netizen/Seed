"""Product-shell boundary guards for retiring the Legacy Neuroplex runtime."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


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
        "api/training/publish.py",
    )
    for relative in modules:
        imports = _imports(REPO / relative)
        assert "seed_platform.paths" in imports, relative
        assert "neuroplex.core.utils" not in imports, relative

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"seed_platform*"' in pyproject


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
    assert "neuroplex.services.tool_service" in source


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
    # .codex/ 是本地 CLI 的 git worktree/临时文件（已被 .gitignore 忽略，不进 CI，
    # 但会在本地 rglob 命中）；.venv* 是本地虚拟环境。两者都不是源码守卫对象。
    skip_parts = {".git", "node_modules", "build", "dist", "_libs", ".venv", ".venv310", ".codex"}
    scanned = 0
    offenders: list[str] = []
    for path in REPO.rglob("*.py"):
        if any(part in skip_parts for part in path.parts):
            continue
        relative = path.relative_to(REPO).as_posix()
        if relative.startswith("scripts/archive/"):
            continue
        scanned += 1
        if path.read_bytes().startswith(b"\xef\xbb\xbf"):
            offenders.append(relative)

    assert scanned > 100, f"BOM 扫描面异常收窄，仅扫到 {scanned} 个文件"
    assert offenders == [], f"以下 Python 源码带 UTF-8 BOM：{offenders}"
