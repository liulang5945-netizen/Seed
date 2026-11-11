"""Contract guard for the 10.7a replay card: its comparison and status logic.

The replay itself needs the model (seconds of CPU), so only the pure parts are
pinned here: which differences count as bitwise, which as tolerance, and which
status wins when several conditions fail at once.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "verify_taiji_training_replay_10_7a.py"


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


REPLAY = _load("verify_taiji_training_replay_10_7a", SCRIPT)
HITS = [False, True, False, True, False, True]


def _rows(hits, shift: float = 0.0):
    return [
        {
            "correct_verdict": -0.140 + index * 1e-3 + shift,
            "other_verdict": -0.141 + index * 1e-3 + shift,
            "hit": hit,
        }
        for index, hit in enumerate(hits)
    ]


def test_identical_replay_is_bitwise():
    stored = {"held_out": _rows(HITS), "held_out_hits": 3}
    replay = {"rows": _rows(HITS), "hits": 3}
    out = REPLAY.compare_arm(stored, replay)
    assert out["bitwise_equal"] is True
    assert out["hits_match"] is True
    assert out["max_abs_delta"] == 0.0


def test_tolerance_shift_is_not_bitwise():
    stored = {"held_out": _rows(HITS), "held_out_hits": 3}
    replay = {"rows": _rows(HITS, shift=1e-12), "hits": 3}
    out = REPLAY.compare_arm(stored, replay)
    assert out["bitwise_equal"] is False
    assert out["hits_match"] is True
    assert 1e-12 <= out["max_abs_delta"] <= 1.001e-12


def test_hit_flip_is_detected():
    stored = {"held_out": _rows(HITS), "held_out_hits": 3}
    flipped = [not hit for hit in HITS]
    replay = {"rows": _rows(flipped), "hits": 3}
    out = REPLAY.compare_arm(stored, replay)
    assert out["hits_match"] is False


def test_short_replay_is_detected():
    stored = {"held_out": _rows(HITS), "held_out_hits": 3}
    replay = {"rows": _rows(HITS[:5]), "hits": 3}
    out = REPLAY.compare_arm(stored, replay)
    assert out["hits_match"] is False
    assert out["rows"] == 5


def test_missing_value_is_not_bitwise():
    stored = {"held_out": _rows(HITS), "held_out_hits": 3}
    replay = {"rows": _rows(HITS), "hits": 3}
    replay["rows"][0]["correct_verdict"] = None
    out = REPLAY.compare_arm(stored, replay)
    assert out["hits_match"] is False
    assert out["bitwise_equal"] is False


def test_status_table():
    status = REPLAY.status_from
    assert (
        status(files_ok=True, replay_ok=True, tolerance_ok=True, budget_ok=True)
        == "weights_replay_bitwise"
    )
    assert (
        status(files_ok=True, replay_ok=False, tolerance_ok=True, budget_ok=True)
        == "weights_replay_within_tolerance"
    )
    assert (
        status(files_ok=True, replay_ok=False, tolerance_ok=False, budget_ok=True)
        == "weights_replay_mismatch"
    )
    assert (
        status(files_ok=False, replay_ok=False, tolerance_ok=False, budget_ok=True)
        == "engineering_failure"
    )
    assert (
        status(files_ok=False, replay_ok=True, tolerance_ok=True, budget_ok=False) == "budget_stop"
    )
