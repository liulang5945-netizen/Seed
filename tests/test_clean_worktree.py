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
