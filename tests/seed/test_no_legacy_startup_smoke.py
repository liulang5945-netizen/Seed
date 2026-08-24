from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


_SMOKE_SCRIPT = r"""
import importlib.abc
import os
import sys


class LegacyImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "neuroplex" or fullname.startswith("neuroplex."):
            raise ModuleNotFoundError(f"blocked legacy import: {fullname}")
        return None


os.environ["SEED_ENABLE_LEGACY"] = "0"
sys.meta_path.insert(0, LegacyImportBlocker())

from api.app import create_app
from api.seed_runtime import DEFAULT_CHECKPOINT, is_seed_active


app = create_app(startup_tasks=False)
paths = app.openapi()["paths"]
assert "/api/health" in paths
assert "/api/runtime/bootstrap" in paths
assert "/api/mcp/marketplace" not in paths
assert "/api/agent/memory/status" not in paths
assert "/api/workspace/path" not in paths
assert DEFAULT_CHECKPOINT.name == "seed_corpus.pt"
assert is_seed_active() is False
print(f"no-legacy startup smoke passed: {len(paths)} API paths")
"""


_LEGACY_SCRIPT = r"""
import os


os.environ["SEED_ENABLE_LEGACY"] = "1"

from api.app import create_app
from api.legacy_bridge import legacy_available


assert legacy_available() is True
app = create_app(startup_tasks=False)
paths = app.openapi()["paths"]
assert "/api/health" in paths
assert "/api/runtime/bootstrap" in paths
assert "/api/life/status" in paths
print(f"legacy startup smoke passed: {len(paths)} API paths")
"""


def test_api_starts_without_legacy_imports():
    env = os.environ.copy()
    env["SEED_ENABLE_LEGACY"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", _SMOKE_SCRIPT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no-legacy startup smoke passed" in result.stdout


def test_api_starts_with_legacy_enabled():
    env = os.environ.copy()
    env["SEED_ENABLE_LEGACY"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _LEGACY_SCRIPT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "legacy startup smoke passed" in result.stdout
