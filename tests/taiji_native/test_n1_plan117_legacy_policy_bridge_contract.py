from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "_taiji_n1_coupled_rwl_legacy_source_loader.py"


def _module():
    spec = importlib.util.spec_from_file_location("n1_plan117_legacy_policy_bridge", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_zero_uses_old_policy_temporarily_then_restores_coupled_policy() -> None:
    module = _module()
    old_policy = {"schema": "legacy"}
    coupled_policy = {"schema": "coupled"}
    observations = []

    def loader(payload):
        observations.append((payload["schema"], dict(coupled_policy)))
        return "loaded"

    result = module.load_source(
        {"schema": module.LEGACY_ZERO_SCHEMA}, loader, coupled_policy, old_policy
    )

    assert result == "loaded"
    assert observations == [(module.LEGACY_ZERO_SCHEMA, old_policy)]
    assert coupled_policy == {"schema": "coupled"}


def test_new_checkpoint_schema_never_enters_legacy_policy_bridge() -> None:
    module = _module()
    coupled_policy = {"schema": "coupled"}
    seen = []

    def loader(payload):
        seen.append(dict(coupled_policy))
        return "restored"

    result = module.load_source(
        {"schema": "n1-coupled-rwl-training-checkpoint-v1"},
        loader,
        coupled_policy,
        {"schema": "legacy"},
    )

    assert result == "restored"
    assert seen == [{"schema": "coupled"}]
    assert coupled_policy == {"schema": "coupled"}
