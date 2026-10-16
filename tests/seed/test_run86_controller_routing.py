from __future__ import annotations

import runpy
from pathlib import Path

from scripts.training.run_taiji_n1_transition_factorial_paired_run86 import (
    install_runner_globals,
    run86c_external_path,
)


def test_controller_override_reaches_functions_in_runpy_globals(tmp_path):
    runner = tmp_path / "frozen_runner.py"
    runner.write_text(
        "TARGET = 'frozen'\ndef readiness():\n    return TARGET\n"
        "def known_views():\n    return TARGET\ndef train_one():\n    return TARGET\n"
        "def identity():\n    return TARGET\ndef audit():\n    return TARGET\n",
        encoding="utf-8",
    )
    module = runpy.run_path(str(runner), run_name="_frozen_runner")

    assert module is not module["audit"].__globals__
    namespace = install_runner_globals(module, {"TARGET": "run86c"})

    assert namespace is module["audit"].__globals__
    assert module["audit"]() == "run86c"
    assert module["readiness"]() == "run86c"
    assert module["known_views"]() == "run86c"
    assert module["train_one"]() == "run86c"


def test_controller_routes_frozen_runner_checkpoints_to_run86c(tmp_path):
    path = run86c_external_path(lambda relative: str(tmp_path / relative), "data/checkpoints")

    assert path == tmp_path / "data/checkpoints" / "run86c"


def test_runpy_overrides_reach_frozen_batch_guard_and_checkpoint_path(tmp_path):
    root = Path(__file__).resolve().parents[2]
    runner = root / "scripts/training/run_taiji_n1_transition_factorial_paired.py"
    module = runpy.run_path(str(runner), run_name="_run86c_frozen_runner_test")
    report = tmp_path / "reports/run86c.json"
    attempt = tmp_path / "reports/run86c_attempt.json"
    checkpoint_root = tmp_path / "data/checkpoints/run86c"
    namespace = install_runner_globals(
        module,
        {
            "REPORT": report,
            "ATTEMPT": attempt,
            "get_external_path": lambda relative: str(tmp_path / relative / "run86c"),
        },
    )

    assert namespace["REPORT"] == report
    assert namespace["ATTEMPT"] == attempt
    assert Path(namespace["get_external_path"]("data/checkpoints")) == checkpoint_root
    assert module["audit"].__globals__ is namespace
