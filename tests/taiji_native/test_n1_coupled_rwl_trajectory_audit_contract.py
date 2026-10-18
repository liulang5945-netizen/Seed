from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_n1_coupled_rwl_trajectory.py"


def _module():
    spec = importlib.util.spec_from_file_location("n1_coupled_rwl_trajectory_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parent_comparison_counts_rescues_regressions_and_joint_errors() -> None:
    module = _module()

    result = module.compare_parent_sets(
        {"original": {"a", "b"}, "rename": {"a"}},
        {"original": {"b", "c"}, "rename": {"a", "d"}},
    )

    assert result["original"] == {
        "n1_110_joint_four": 2,
        "n1_117_joint_four": 2,
        "unchanged_correct": 1,
        "rescued_by_n1_117": ["c"],
        "lost_from_n1_110": ["a"],
        "both_wrong": 45,
    }
    assert result["rename"]["rescued_by_n1_117"] == ["d"]
    assert result["rename"]["lost_from_n1_110"] == []


def test_trajectory_audit_is_offline_and_never_loads_a_model_or_checkpoint() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not imports.intersection({"torch", "taiji"})
    assert not calls.intersection({"load_state_dict", "load_model", "measure", "generate", "save"})
    assert 'checkpoint_payload_reads": 0' in SCRIPT.read_text(
        encoding="utf-8"
    ) or '"checkpoint_payload_reads": 0' in SCRIPT.read_text(encoding="utf-8")
