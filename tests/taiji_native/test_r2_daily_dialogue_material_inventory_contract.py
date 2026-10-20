from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts.data_prep.audit_taiji_r2_daily_dialogue_material_inventory import (
    _episode_metadata,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/data_prep/audit_taiji_r2_daily_dialogue_material_inventory.py"


def test_inventory_entrypoint_imports_from_outside_the_repository(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SCRIPT.parent)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1], run_name='inventory_import_check')",
            str(SCRIPT),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_episode_metadata_parser_handles_default_json_spacing_without_reading_target() -> None:
    line = (
        '{"episode_id": "e1", "family_id": "tree-1", "task_family": "daily_dialogue", '
        '"split": "train", "user_input": "你好", "response": "PRIVATE TARGET"}'
    )

    assert _episode_metadata(line) == ("tree-1", "train")
