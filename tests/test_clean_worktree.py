from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLEANER = REPO / "scripts" / "clean_worktree.py"


def _module():
    spec = importlib.util.spec_from_file_location("clean_worktree_under_test", CLEANER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_empty_directory_cleanup_preserves_protected_and_historical_roots(tmp_path) -> None:
    module = _module()
    module.ROOT = str(tmp_path)
    protected = module.PRESERVE_EMPTY_ROOTS
    for root_name in protected:
        (tmp_path / root_name / "empty-child").mkdir(parents=True)
    (tmp_path / "safe-cache" / "empty-child").mkdir(parents=True)

    found = {path.replace("\\", "/") for path in module.plan_empty_dirs(apply=False)}

    assert found == {"safe-cache/empty-child"}


def test_empty_directory_cleanup_keeps_directories_with_tracked_history(tmp_path) -> None:
    module = _module()
    module.ROOT = str(tmp_path)
    (tmp_path / "direct-history" / "empty-child").mkdir(parents=True)
    (tmp_path / "ordinary-cache" / "empty-child").mkdir(parents=True)

    found = {path.replace("\\", "/") for path in module.plan_empty_dirs(apply=False)}

    assert found == {"ordinary-cache/empty-child"}


def test_remove_dir_retries_readonly_runtime_files(tmp_path) -> None:
    module = _module()
    module.ROOT = str(tmp_path)
    target = tmp_path / "scratch" / "readonly.bin"
    target.parent.mkdir()
    target.write_bytes(b"generated test cache")
    target.chmod(stat.S_IREAD)

    assert module.remove_dir("scratch") is True
    assert not target.parent.exists()


def test_pytest_basetemp_families_are_included_but_cache_is_not(tmp_path, monkeypatch) -> None:
    module = _module()
    module.ROOT = str(tmp_path)
    for name in (
        ".pytest-tmp-n1-run100-contracts",
        ".pytest-final-all",
        ".pytest_cache",
    ):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(module, "git_ignored", lambda _rel: True)

    planned = {rel for rel, _count, _mb, verdict in module.plan_dirs() if verdict == "删除"}

    assert planned == {".pytest-tmp-n1-run100-contracts", ".pytest-final-all"}


def test_default_cleanup_keeps_state_overlays_and_dependencies(tmp_path, monkeypatch) -> None:
    module = _module()
    module.ROOT = str(tmp_path)
    payloads = []
    for name in (".seed_test_tmp", "_libs", "logs", "rag_data", "user_data", "agent_workspace"):
        payload = tmp_path / name / "retained-state.txt"
        payload.parent.mkdir()
        payload.write_text("unique state or source overlay", encoding="utf-8")
        payloads.append(payload)
    build = tmp_path / "build"
    build.mkdir()
    (build / "generated.txt").write_text("reproducible", encoding="utf-8")
    monkeypatch.setattr(module, "git_ignored", lambda _rel: True)

    planned = [rel for rel, _n, _mb, verdict in module.plan_dirs() if verdict == "删除"]
    assert planned == ["build"]
    for rel in planned:
        assert module.remove_dir(rel)
    assert not build.exists()
    assert all(
        path.read_text(encoding="utf-8") == "unique state or source overlay" for path in payloads
    )
