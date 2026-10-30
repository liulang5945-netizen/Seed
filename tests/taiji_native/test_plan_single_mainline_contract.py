"""The plan tree keeps exactly one declared mainline and queue head."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/training/audit_plan_single_mainline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_plan_single_mainline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_plan_tree_passes_single_mainline_audit():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok", payload["failures"]
    assert payload["rc"] == 0


def test_guard_rejects_a_second_queue_head(tmp_path):
    module = _load_module()
    plans = tmp_path / "plans"
    (plans / "active/roadmap").mkdir(parents=True)
    (plans / "reference").mkdir(parents=True)
    (plans / "MAINLINE.md").write_text(
        "# 唯一主线\n\n> 文档状态：[主线]\n\n唯一执行队列：R3b\n", encoding="utf-8"
    )
    (plans / "README.md").write_text("[MAINLINE](MAINLINE.md)\n", encoding="utf-8")
    (plans / "active/PLAN_INDEX.md").write_text(
        "[03](roadmap/03_CURRENT_EXECUTION.md)\n[01](roadmap/01_SCOPE_AND_PHASES.md)\n",
        encoding="utf-8",
    )
    (plans / "active/roadmap/03_CURRENT_EXECUTION.md").write_text(
        "# 03\n\n> 文档状态：[状态]\n\n## 当前唯一下一步：唯一\n", encoding="utf-8"
    )
    (plans / "active/roadmap/01_SCOPE_AND_PHASES.md").write_text(
        "# 01\n\n> 文档状态：[历史]\n\n## 当前唯一下一步：第二套\n", encoding="utf-8"
    )
    (plans / "reference/README.md").write_text(
        "[LEGACY_ASSET_INDEX](LEGACY_ASSET_INDEX.md)\n", encoding="utf-8"
    )
    (plans / "reference/LEGACY_ASSET_INDEX.md").write_text("# 旧资产\n", encoding="utf-8")
    for relative in module.CURRENT_CONTRACTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 合同\n\n> 文档状态：[合同]\n", encoding="utf-8")
    for relative in module.LEGACY_BANNER_REQUIRED:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 旧资产\n\n> 文档状态：[历史]\n", encoding="utf-8")
    result = module.audit(tmp_path)
    assert result["status"] == "failed"
    assert any("only 03_CURRENT_EXECUTION.md" in item for item in result["failures"])
