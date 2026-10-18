from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "scripts" / "training" / "audit_taiji_n1_position_five_group_itemwise_interactions.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("n1_plan116_itemwise_interactions", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pair_interaction_is_zero_for_additive_effects_and_exact_for_joint_effect() -> None:
    module = _module()

    assert module.interaction(10, 12, 15, 17) == 0
    assert module.interaction(10, 12, 15, 20) == 3


def test_itemwise_audit_has_no_model_or_checkpoint_mutation_path() -> None:
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
    assert not calls.intersection({"measure", "generate", "backward", "step", "save"})
