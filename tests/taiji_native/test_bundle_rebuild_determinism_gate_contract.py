"""Contract for the rebuild-determinism gate (ledger D-20261111-03).

The gate decides one thing: two rebuilds of the 09.8 bundle, produced by two
separate processes in the same job, read out the same.  These tests pin the
three exits and the line between a reading and a diagnostic, so the boundary
cannot drift without a red test:

* diagnostics (timings, memory, shas, byte counts, provenance) never decide;
* one unit in the last place is allowed on floating layers, two are not;
* structure means keys, lengths, dtypes and shapes;
* a missing runtime condition makes the verdict ``indeterminate``, never green.

They do not rebuild anything: the build path is exercised by the step that runs
this gate, and repeating it here would add ~20 minutes to every suite run.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import verify_taiji_bundle_rebuild_determinism as gate  # noqa: E402


def test_diagnostics_never_decide_equivalence() -> None:
    left = {
        "status": "passed",
        "rows": [{"mass": 0.25}],
        "cost": {"cpu_seconds": 1.0},
        "elapsed_seconds": 2.0,
        "sha256": "a" * 64,
        "bytes": 77609636,
        "identity": {"bundle_sha256": "a" * 64},
        "report": "reports/a.json",
    }
    right = {
        "status": "passed",
        "rows": [{"mass": 0.25}],
        "cost": {"cpu_seconds": 99.0},
        "elapsed_seconds": 42.0,
        "sha256": "b" * 64,
        "bytes": 1,
        "identity": {"bundle_sha256": "b" * 64},
        "report": "reports/b.json",
    }

    assert gate.compare_readouts(left, right) == []


def test_nested_timings_and_autograd_nodes_are_diagnostics() -> None:
    """Consumers nest timings inside arms/checks; listing top-level names is not enough."""
    left = {
        "arms": [{"eval_score_cpu": 0.2265625, "generation_cpu": 12.0, "score": 0.5}],
        "checks": [{"observed": {"b_goal_grad_fn": "<TanhBackward0 object at 0x1>"}}],
        "cpu_seconds_including_prior_attempts": 27.484375,
        "estimated_wall_seconds": 12.5,
        "payload_dir": "output/a",
    }
    right = {
        "arms": [{"eval_score_cpu": 0.23828125, "generation_cpu": 99.0, "score": 0.5}],
        "checks": [{"observed": {"b_goal_grad_fn": "<TanhBackward0 object at 0x2>"}}],
        "cpu_seconds_including_prior_attempts": 27.609375,
        "estimated_wall_seconds": 42.5,
        "payload_dir": "output/b",
    }

    assert gate.compare_readouts(left, right) == []


def test_reading_names_are_not_swallowed_by_the_diagnostic_rule() -> None:
    for reading in ("target_mass", "score", "mass_ratio", "recalled_norm", "passed", "rows"):
        assert not gate.is_diagnostic(reading), reading


def test_a_reading_that_moves_is_a_problem() -> None:
    assert gate.compare_readouts({"status": "passed"}, {"status": "engineering_failure"})
    assert gate.compare_readouts({"rows": 15}, {"rows": 14})
    assert gate.compare_readouts({"checks": [True]}, {"checks": [False]})


def test_readout_floats_use_a_relative_tolerance() -> None:
    within = gate.compare_readouts({"mass": 0.5}, {"mass": 0.5 + 1e-13})
    beyond = gate.compare_readouts({"mass": 0.5}, {"mass": 0.5 + 1e-9})

    assert within == []
    assert beyond and beyond[0]["issue"] == "float"


def test_nan_readouts_agree_with_each_other() -> None:
    assert gate.compare_readouts({"mass": float("nan")}, {"mass": float("nan")}) == []


def test_structure_pins_keys_lengths_dtypes_and_shapes() -> None:
    left = {"parameters": {"w": torch.zeros(2, dtype=torch.float64)}, "history": [1, 2]}
    right_ok = {"parameters": {"w": torch.zeros(2, dtype=torch.float64)}, "history": [1, 2]}
    right_missing = {"parameters": {}, "history": [1, 2]}
    right_dtype = {"parameters": {"w": torch.zeros(2, dtype=torch.float32)}, "history": [1, 2]}
    right_shape = {"parameters": {"w": torch.zeros(3, dtype=torch.float64)}, "history": [1, 2]}
    right_length = {"parameters": {"w": torch.zeros(2, dtype=torch.float64)}, "history": [1]}

    assert gate.compare_structure(left, right_ok) == []
    assert {row["issue"] for row in gate.compare_structure(left, right_missing)} == {"missing"}
    assert gate.compare_structure(left, right_dtype)[0]["issue"] == "dtype"
    assert gate.compare_structure(left, right_shape)[0]["issue"] == "shape"
    assert gate.compare_structure(left, right_length)[0]["issue"] == "length"


def test_numbers_allow_one_unit_in_the_last_place() -> None:
    one = torch.tensor([1.0], dtype=torch.float64)
    neighbor = torch.nextafter(one, torch.tensor([2.0], dtype=torch.float64))
    two_ulp = torch.nextafter(neighbor, torch.tensor([2.0], dtype=torch.float64))

    assert gate.compare_numbers({"w": one}, {"w": neighbor})["failures"] == []
    assert gate.compare_numbers({"w": one}, {"w": two_ulp})["failures"]


def test_numbers_allow_one_ulp_on_float32_too() -> None:
    one = torch.tensor([1.0], dtype=torch.float32)
    neighbor = torch.nextafter(one, torch.tensor([2.0], dtype=torch.float32))
    two_ulp = torch.nextafter(neighbor, torch.tensor([2.0], dtype=torch.float32))

    assert gate.compare_numbers({"w": one}, {"w": neighbor})["failures"] == []
    assert gate.compare_numbers({"w": one}, {"w": two_ulp})["failures"]


def test_integer_tensors_must_be_bit_identical() -> None:
    left = {"step": torch.tensor([7])}
    assert gate.compare_numbers(left, {"step": torch.tensor([7])})["failures"] == []
    assert gate.compare_numbers(left, {"step": torch.tensor([8])})["failures"]


@pytest.mark.parametrize(
    ("build_exit", "structure", "numbers", "readout_indeterminate", "readout_problems", "verdict"),
    [
        (0, [], [], False, [], "equivalent"),
        (0, [], [], False, [{"path": "/10_21_1/status", "issue": "value"}], "not_equivalent"),
        (0, [{"path": "/parameters", "issue": "shape"}], [], False, [], "not_equivalent"),
        (0, [], [{"path": "/parameters/w"}], False, [], "not_equivalent"),
        (0, [], [], True, [], "indeterminate"),
        (1, [], [], False, [], "indeterminate"),
    ],
)
def test_verdict_has_exactly_three_exits(
    build_exit: int,
    structure: list,
    numbers: list,
    readout_indeterminate: bool,
    readout_problems: list,
    verdict: str,
) -> None:
    builds = []
    for label in ("a", "b"):
        row = {"label": label, "exit": build_exit}
        if build_exit == 0:
            row["sha256"] = "0" * 64
        builds.append(row)
    layer = {
        "structure": {"problems": structure},
        "numbers": {"failures": numbers},
        "readouts": {"indeterminate": readout_indeterminate, "problems": readout_problems},
    }

    assert gate.summarise(builds, layer) == verdict


def test_missing_runtime_conditions_are_named() -> None:
    assert gate.missing_conditions({}) == [
        "python",
        "platform",
        "torch",
        "torch_threads",
        "torch_interop_threads",
        "isa",
        "blas",
    ]
    assert (
        gate.missing_conditions(
            {
                "python": "3.12.10",
                "platform": "Windows",
                "torch": "2.13.0+cpu",
                "torch_threads": 1,
                "torch_interop_threads": 1,
                "isa": "AVX512",
                "blas": "mkl",
            }
        )
        == []
    )


def test_the_gate_conditions_are_recorded_by_the_gate_itself() -> None:
    conditions = gate.runtime_conditions()
    assert gate.missing_conditions(conditions) == []


def test_diagnostic_names_do_not_shadow_readings() -> None:
    for reading in ("status", "rows", "checks", "prices", "arms", "readings", "passed"):
        assert reading not in gate.DIAGNOSTIC_KEYS


def test_readouts_clear_stale_reports_before_consumers_run(tmp_path, monkeypatch) -> None:
    """一个失败的消费者不许被上一轮留下的报告冒充成"有报告"。

    消费者都是 write-once（present report ⇒ refuse）。这里用一个同样行为的假消费者：
    只要输出路径已存在就 exit 3。如果 ``run_readouts`` 没有先清目录，残留的
    ``fake.json`` 会让假消费者拒写、报告保持旧内容，判定就变成拿旧读数比新读数——
    与 ``SUPERSEDED`` 那批旧报告冒充新报告的缺陷同族。
    """
    consumer = tmp_path / "write_once_consumer.py"
    consumer.write_text(
        "import json, os, sys\n"
        "path = sys.argv[sys.argv.index('--output') + 1]\n"
        "if os.path.exists(path):\n"
        "    print('refusing to overwrite ' + path, file=sys.stderr)\n"
        "    sys.exit(3)\n"
        "with open(path, 'w', encoding='utf-8') as handle:\n"
        "    json.dump({'status': 'passed'}, handle)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gate,
        "readouts",
        lambda: {
            "fake": {
                "script": str(consumer),
                "report": "fake.json",
                "args": ("--output", "{report}"),
                "timeout": 60,
            }
        },
    )
    work = tmp_path / "work"
    stale_dir = work / "readouts_a"
    stale_dir.mkdir(parents=True)
    stale = stale_dir / "fake.json"
    stale.write_text('{"status": "stale"}', encoding="utf-8")

    results = gate.run_readouts(tmp_path / "bundle.pt", "a", work)

    assert results["fake"]["exit"] == 0, results["fake"]["stderr_tail"]
    assert results["fake"]["report_present"] is True
    assert stale.read_text(encoding="utf-8") == '{"status": "passed"}'
