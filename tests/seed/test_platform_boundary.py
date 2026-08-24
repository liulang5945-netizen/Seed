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
    app = REPO / "api" / "app.py"
    imports = _imports(app)

    assert "api.legacy_bridge" in imports
    assert not any(module.startswith("neuroplex") for module in imports)


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
