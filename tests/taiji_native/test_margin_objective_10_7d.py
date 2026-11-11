"""Contract guard for the 10.7d margin probe: the verdict table over structure stats."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "diagnose_margin_objective_10_7d.py"


def _load(name: str, path: Path):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MARGIN = _load("diagnose_margin_objective_10_7d", SCRIPT)


def _stats(fraction, median, coherence):
    return {
        "explained_fraction": fraction,
        "pairwise_cosine_median": median,
        "within_group_cosine": coherence,
    }


def test_margin_isolates():
    assert (
        MARGIN.classify_margin(_stats(0.8, 0.3, {"a": 0.9, "b": 0.8}))
        == "margin_isolates_differential"
    )


def test_margin_still_common():
    assert MARGIN.classify_margin(_stats(0.8, 0.999, {"a": 0.9, "b": 0.9})) == "margin_still_common"


def test_margin_mixed_when_structure_weak():
    assert MARGIN.classify_margin(_stats(0.2, 0.5, {"a": 0.9, "b": 0.9})) == "margin_mixed"
    assert MARGIN.classify_margin(_stats(0.8, 0.5, {"a": 0.1, "b": 0.9})) == "margin_mixed"


def test_margin_undefined():
    assert MARGIN.classify_margin(_stats(0.8, None, {"a": 0.9})) == "margin_undefined"
    assert MARGIN.classify_margin(_stats(None, 0.5, {})) == "margin_undefined"
