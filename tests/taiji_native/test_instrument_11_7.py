"""Contract guard for the 11.7 instrument card: its three pure helpers.

The measurement needs the model, so only the classification, percentile and
status-precedence tables are pinned here -- each with a case that proves it can
say no.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "verify_taiji_instrument_11_7.py"


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


INSTRUMENT = _load("verify_taiji_instrument_11_7", SCRIPT)


def test_read_only_and_pycache_are_allowed():
    assert INSTRUMENT.classify_open("r", "reports/x.json") == "read"
    assert INSTRUMENT.classify_open("rb", "output/x.pt") == "read"
    assert INSTRUMENT.classify_open("w", "pkg/__pycache__/m.cpython-312.pyc") == "pycache_write"
    assert INSTRUMENT.classify_open("wb", "pkg\\__pycache__\\m.pyc") == "pycache_write"


def test_real_writes_are_flagged():
    for mode in ("w", "wb", "a", "x", "r+", "w+"):
        assert INSTRUMENT.classify_open(mode, "reports/x.json") == "write"
    assert INSTRUMENT.classify_open("w", "pkg/m.py") == "write"


def test_nearest_rank():
    values = [3.0, 1.0, 2.0, 4.0, 5.0]
    assert INSTRUMENT.nearest_rank(values, 50) == 3.0
    assert INSTRUMENT.nearest_rank(values, 95) == 5.0
    assert INSTRUMENT.nearest_rank(values, 100) == 5.0
    assert INSTRUMENT.nearest_rank([], 50) is None
    assert INSTRUMENT.nearest_rank([7.0], 50) == 7.0


def test_status_precedence():
    status = INSTRUMENT.status_from
    assert (
        status(budget_ok=True, unsafe=False, nondeterministic=False, failed=False)
        == "instrument_measured_quiet"
    )
    assert (
        status(budget_ok=True, unsafe=False, nondeterministic=False, failed=True)
        == "engineering_failure"
    )
    assert (
        status(budget_ok=True, unsafe=False, nondeterministic=True, failed=False)
        == "nondeterministic_output"
    )
    assert (
        status(budget_ok=True, unsafe=True, nondeterministic=False, failed=True)
        == "unsafe_side_effect"
    )
    assert status(budget_ok=False, unsafe=True, nondeterministic=True, failed=True) == "budget_stop"
