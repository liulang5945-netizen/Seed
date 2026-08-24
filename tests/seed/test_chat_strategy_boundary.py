from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


_SCRIPT = r"""
import importlib.abc
import os
import sys
import asyncio


class LegacyImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "neuroplex" or fullname.startswith("neuroplex."):
            raise ModuleNotFoundError(f"blocked legacy import: {fullname}")
        return None


os.environ["SEED_ENABLE_LEGACY"] = "0"
sys.meta_path.insert(0, LegacyImportBlocker())

from api import chat_strategies
from api.models import ChatRequest
from api.routes_chat import chat_stream


assert chat_strategies._get_life_state() == {}
assert chat_strategies._get_memory_context() == ""
assert chat_strategies._has_react_engine() is False
chat_strategies._record_life_interaction()
chat_strategies._record_evolution("prompt", "answer", True)
chat_strategies._record_recursive_strategies("prompt", "system", True, 0, set())
response = asyncio.run(chat_stream(ChatRequest(prompt="ping")))
assert response.media_type == "text/event-stream"
print("chat strategy legacy gate passed")
"""


def test_chat_strategy_does_not_import_legacy_when_disabled():
    env = os.environ.copy()
    env["SEED_ENABLE_LEGACY"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "chat strategy legacy gate passed" in result.stdout
