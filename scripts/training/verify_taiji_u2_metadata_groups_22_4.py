"""Zero-update verification for U2 22.4.1 metadata entry and group-local growth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_learning_transaction import rebuild_state
from taiji.u2_recovery import ENVELOPE_SCHEMA, U2Runtime
from taiji.u2_subject import U2Config, U2Record, U2Subject

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_1_verification_contract_20261111.json"
SUBJECT_V3 = "taiji-u2-subject-v3"
TOLERANCE = {
    torch.float64: {"atol": 1e-10, "rtol": 1e-8},
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
}


class StepGuard:
    """Hard-block every optimizer update inside the zero-update scope."""

    def __init__(self) -> None:
        self.calls = 0
        self._original = {}

    def __enter__(self) -> StepGuard:
        def blocked(*_args, **_kwargs):
            self.calls += 1
            raise AssertionError("optimizer.step is outside the 22.4.1 zero-update scope")

        for name in ("Adam", "AdamW", "SGD"):
            owner = getattr(torch.optim, name)
            self._original[name] = owner.step
            owner.step = blocked
        return self

    def __exit__(self, *_exc) -> None:
        for name, step in self._original.items():
            getattr(torch.optim, name).step = step


def candidate(precision: str) -> tuple[U2Subject, U2Subject]:
    parent = U2Subject(U2Config.small()).to(dtype=getattr(torch, precision))
    return parent, parent.metadata_candidate()


def prepared(precision: str) -> tuple[U2Subject, U2Runtime]:
    parent, model = candidate(precision)
    runtime = U2Runtime(model)
    runtime.set_goal("红盒现在在哪里？")
    runtime.set_records(
        (
            U2Record("a", "红盒在左边。".encode(), 0, observed_at=-1.0),
            U2Record("b", "红盒在右边。".encode(), 1),
        )
    )
    return parent, runtime


def credit(model: U2Subject, group: int) -> torch.Tensor:
    value = model.native_w.new_zeros(model.c.groups, model.c.d, 8 * model.c.d)
    value[group, 0, 0], value[group, 1, 1] = 3, 2
    return value


def native_value(runtime: U2Runtime) -> torch.Tensor:
    model = runtime.model
    return model.native(
        runtime.state,
        model.goal(runtime.state["goal_raw"]),
        model.native_w.new_zeros(15),
        model.native_w.new_full((model.c.branches, model.c.d), 0.1),
    )


def answer_loss(runtime: U2Runtime) -> torch.Tensor:
    model = runtime.model
    state = model.begin_utterance(runtime.state)
    goal = model.goal(state["goal_raw"])
    terms = []
    for symbol in [*"右边".encode(), 256]:
        probability, _, output = model.output(state, goal, state["prediction_action"])
        terms.append(-probability[symbol])
        state = model.accept_byte(state, symbol, output)
    return torch.stack(terms).mean()


def attach_optimizer(runtime: U2Runtime) -> None:
    model = runtime.model
    rest = [
        value for name, value in model.named_parameters() if name not in {"native_r", "native_p"}
    ]
    runtime.optimizer = torch.optim.Adam(
        [
            {"params": [model.native_r], "lr": 0.001},
            {"params": [model.native_p], "lr": 0.002},
            {"params": rest},
        ],
        lr=0.003,
    )


def successor(runtime: U2Runtime) -> dict[str, str]:
    """One declared successor birth, used identically in both processes."""
    runtime.birth_transaction(credit(runtime.model, 1), groups=(1,), birth_id="successor")
    return {
        "model": value_digest(runtime.model.state_dict()),
        "state": value_digest(runtime.state),
        "structure": value_digest(runtime.model.structure()),
        "optimizer": value_digest(runtime.optimizer.state_dict() if runtime.optimizer else None),
        "native": value_digest(native_value(runtime)),
        "widths": list(runtime.model.c.group_units or ()),
    }


def check_detail(runtime: U2Runtime) -> dict[str, object]:
    return {
        "model": value_digest(runtime.model.state_dict()),
        "state": value_digest(runtime.state),
        "structure": value_digest(runtime.model.structure()),
        "optimizer": value_digest(runtime.optimizer.state_dict() if runtime.optimizer else None),
    }


def worker(envelope: Path, destination: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    started = time.process_time()
    with StepGuard() as guard:
        runtime = U2Runtime.load(envelope)
        rebuilt, _ = rebuild_state(runtime.model, runtime.history, runtime.state)
        assert_values_equal(runtime.state, rebuilt)
        loaded = check_detail(runtime)
        advanced = successor(runtime)
    payload = {
        "loaded": loaded,
        "successor": advanced,
        "optimizer_steps": guard.calls,
        "threads": torch.get_num_threads(),
        "cpu_seconds": time.process_time() - started,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run(report: Path) -> int:
    started = time.process_time()
    results: list[dict[str, object]] = []
    failure: str | None = None
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    expected = [item["id"] for item in contract["criteria"]]

    def record(name: str, passed: bool, **detail: object) -> None:
        results.append({"id": name, "passed": passed, "detail": detail})

    payload: dict[str, object] = {}
    try:
        torch.set_num_threads(1)
        with StepGuard() as guard, tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # 1. Schema pin: the envelope stays v3; only the subject bundle advances.
            baseline = U2Runtime(U2Subject(U2Config.small()))
            baseline_bundle = baseline.model.export_bundle(baseline.state)
            _, metadata_runtime = prepared("float64")
            metadata_bundle = metadata_runtime.model.export_bundle(metadata_runtime.state)
            record(
                "envelope_schema_pin",
                baseline.to_envelope()["schema"] == ENVELOPE_SCHEMA
                and metadata_runtime.to_envelope()["schema"] == ENVELOPE_SCHEMA
                and baseline_bundle["schema"] == "taiji-u2-subject-v2"
                and metadata_bundle["schema"] == SUBJECT_V3,
                envelope=ENVELOPE_SCHEMA,
                baseline_subject=baseline_bundle["schema"],
                metadata_subject=metadata_bundle["schema"],
            )

            # 2-6. Per-precision graph checks.
            forks: dict[str, str] = {}
            gradients: dict[str, int] = {}
            pairs: dict[str, object] = {}
            for precision in ("float64", "float32"):
                parent, model = candidate(precision)
                values = model.state_dict()
                inherited = all(
                    torch.equal(tensor, values[name])
                    for name, tensor in parent.state_dict().items()
                )
                forks[precision] = (
                    f"parent={len(list(parent.parameters()))},"
                    f"candidate={len(list(model.parameters()))},"
                    f"revision={parent.parameter_revision}->{model.parameter_revision},"
                    f"inherited={inherited}"
                )
                assert inherited
                assert len(list(parent.parameters())) == 191
                assert len(list(model.parameters())) == 194
                assert model.parameter_revision == parent.parameter_revision + 1

                _, runtime = prepared(precision)
                goal = runtime.model.goal(runtime.state["goal_raw"])
                before = runtime.model.record_read(runtime.state, goal)
                records = runtime.state["records"]
                versioned = runtime.model.set_records(
                    runtime.state, (replace(records[0], version=1), replace(records[1], version=0))
                )
                after = runtime.model.record_read(versioned, goal)
                units_before = runtime.model.record_units(runtime.state)[0]
                timed = runtime.model.set_records(
                    runtime.state, (replace(records[0], observed_at=-2.0), records[1])
                )
                units_after = runtime.model.record_units(timed)[0]
                raw_equal = [row.raw for row in records] == [row.raw for row in timed["records"]]
                pairs[precision] = {
                    "H_equal": torch.equal(runtime.state["H"], versioned["H"]),
                    "F_equal": torch.equal(runtime.state["F"], versioned["F"]),
                    "content_equal": torch.equal(before[3], after[3]),
                    "route_differs": not torch.equal(before[1], after[1]),
                    "time_units_differ": not torch.equal(units_before, units_after),
                    "raw_equal": raw_equal,
                }
                assert all(
                    value for key, value in pairs[precision].items() if key != "route_differs"
                )
                assert pairs[precision]["route_differs"]

                _, runtime = prepared(precision)
                parameters = [
                    value
                    for name, value in runtime.model.named_parameters()
                    if name.startswith(("record_metadata.", "record_provenance."))
                ]
                measured = torch.autograd.grad(answer_loss(runtime), parameters)
                assert len(measured) == 3
                assert all(
                    bool(torch.isfinite(item).all()) and float(item.norm()) > 0 for item in measured
                )
                gradients[precision] = 3
            record("metadata_fork_inherits_parent", True, precisions=forks)
            record("metadata_pairs", True, precisions=pairs)
            record("metadata_task_gradient", True, tensors=gradients)

            # 7-10. Group growth, neutrality, optimizer ownership and round-trip.
            parent, runtime = prepared("float64")
            model = runtime.model
            width_unit = 9 * model.c.d
            base_snapshot = {name: value.clone() for name, value in model.state_dict().items()}
            neutral_before = native_value(runtime).detach()
            count_before = sum(value.numel() for value in model.parameters())
            attach_optimizer(runtime)
            runtime.birth_transaction(credit(model, 1), groups=(1,), birth_id="second-group-first")
            model = runtime.model
            assert sum(value.numel() for value in model.parameters()) - count_before == width_unit
            runtime.birth_transaction(credit(model, 0), groups=(0,), birth_id="first-group-next")
            model = runtime.model
            extension = model.group_extensions["0"]
            runtime.optimizer.state[extension.incoming] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.full_like(extension.incoming, 0.2),
                "exp_avg_sq": torch.full_like(extension.incoming, 0.3),
            }
            runtime.optimizer.state[extension.outgoing] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.full_like(extension.outgoing, 0.4),
                "exp_avg_sq": torch.full_like(extension.outgoing, 0.5),
            }
            runtime.birth_transaction(credit(model, 0), groups=(0,), birth_id="first-group-again")
            model = runtime.model
            extension = model.group_extensions["0"]
            incoming = runtime.optimizer.state[extension.incoming]
            outgoing = runtime.optimizer.state[extension.outgoing]
            old_blocks = (
                torch.equal(incoming["exp_avg"][:1], torch.full_like(incoming["exp_avg"][:1], 0.2))
                and not bool(incoming["exp_avg"][1:].any())
                and torch.equal(
                    outgoing["exp_avg"][:, :1], torch.full_like(outgoing["exp_avg"][:, :1], 0.4)
                )
                and not bool(outgoing["exp_avg"][:, 1:].any())
                and float(incoming["step"]) == float(outgoing["step"]) == 3
            )
            owners = any(
                value is extension.incoming for value in runtime.optimizer.param_groups[0]["params"]
            ) and any(
                value is extension.outgoing for value in runtime.optimizer.param_groups[1]["params"]
            )
            untouched = all(
                torch.equal(value, model.state_dict()[name])
                for name, value in base_snapshot.items()
            )
            widths = model.c.group_units or ()
            record(
                "group_two_generations",
                widths == (14, 13) and untouched,
                widths=list(widths),
                unit_parameters=width_unit,
                lineage=[len(row) for row in model.unit_ids],
                base_unchanged=untouched,
            )
            neutral_after = native_value(runtime).detach()
            neutral = torch.allclose(neutral_before, neutral_after, **TOLERANCE[torch.float64])
            record(
                "group_neutral_output",
                neutral,
                tolerance=TOLERANCE[torch.float64],
                max_abs_delta=float((neutral_before - neutral_after).abs().max()),
            )
            record(
                "optimizer_owner_moments",
                owners and old_blocks,
                groups=[len(group["params"]) for group in runtime.optimizer.param_groups],
                owners=owners,
                moments=old_blocks,
            )

            envelope = directory / "grouped.bin"
            runtime.save(envelope)
            envelope_sha = hashlib.sha256(envelope.read_bytes()).hexdigest()
            replay = U2Runtime.load(envelope)
            rebuilt, _ = rebuild_state(replay.model, replay.history, replay.state)
            roundtrip = {
                "model": value_digest(runtime.model.state_dict())
                == value_digest(replay.model.state_dict()),
                "state": value_digest(runtime.state) == value_digest(replay.state),
                "structure": value_digest(runtime.model.structure())
                == value_digest(replay.model.structure()),
                "optimizer": value_digest(runtime.optimizer.state_dict())
                == value_digest(replay.optimizer.state_dict()),
                "rebuild": value_digest(rebuilt) == value_digest(replay.state),
            }
            record("envelope_roundtrip_rebuild", all(roundtrip.values()), **roundtrip)

            advanced = successor(replay)
            child_out = directory / "worker.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(envelope),
                    "--out",
                    str(child_out),
                ],
                cwd=ROOT,
                check=True,
                env=dict(os.environ),
            )
            child = json.loads(child_out.read_text(encoding="utf-8"))
            independent = child["successor"] == advanced and child["loaded"] == check_detail(
                U2Runtime.load(envelope)
            )
            record(
                "independent_process_successor",
                independent,
                in_process=advanced,
                child=child["successor"],
                child_optimizer_steps=child["optimizer_steps"],
                child_cpu_seconds=child["cpu_seconds"],
            )

            restored_model, _ = U2Subject.from_bundle(baseline_bundle)
            legacy = len(list(restored_model.parameters())) == 191 and value_digest(
                restored_model.state_dict()
            ) == value_digest(baseline.model.state_dict())
            record(
                "legacy_bundle_compat",
                legacy,
                schema=baseline_bundle["schema"],
                parameters=len(list(restored_model.parameters())),
            )

            payload = {
                "checkpoint_sha256": envelope_sha,
                "checkpoint_bytes": envelope.stat().st_size,
                "group_units": list(model.c.group_units or ()),
                "asset_head": value_digest(model.state_dict()),
            }
            record("zero_optimizer_steps", guard.calls == 0, calls=guard.calls)
            assert guard.calls == 0
    except Exception as error:  # noqa: BLE001 - the report must carry the failure
        failure = repr(error)

    produced = [item["id"] for item in results]
    missing = sorted(set(expected) - set(produced))
    if missing:
        failure = (
            f"missing checks: {missing}" if failure is None else f"{failure}; missing {missing}"
        )
    passed = failure is None and all(item["passed"] for item in results)
    report_payload = {
        "schema": "taiji-u2-22.4.1-metadata-groups-verification-v1",
        "verdict": (
            contract["pass_rule"]["verdict"] if passed else "failed_zero_update_metadata_groups"
        ),
        "contract": str(CONTRACT.relative_to(ROOT)).replace("\\", "/"),
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "parent": contract["parent"],
        "checks": results,
        "failure": failure,
        "optimizer_steps": 0 if passed else None,
        "formal_training": False,
        "project_dataset_reads": 0,
        "check_reads": 0,
        "final_reads": 0,
        "precisions": contract["precisions"],
        "conditions": {
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "artifacts": payload,
        "cpu_seconds": time.process_time() - started,
        "source_pins": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "taiji/u2_record_metadata.py",
                "taiji/u2_group_structure.py",
                "taiji/u2_subject.py",
                "taiji/u2_recovery.py",
                "taiji/u2_learning_transaction.py",
                "scripts/training/verify_taiji_u2_metadata_groups_22_4.py",
            ]
        },
    }
    write_report(report, report_payload)
    print(json.dumps({"verdict": report_payload["verdict"], "checks": len(results)}))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    if arguments.worker is not None:
        if arguments.out is None:
            raise SystemExit("--worker requires --out")
        worker(arguments.worker, arguments.out)
        return 0
    if arguments.report is None:
        raise SystemExit("--report is required")
    target = arguments.report.resolve()
    if target.parent != ROOT / "reports":
        raise ValueError("Report must be directly inside reports/")
    if target.exists():
        raise FileExistsError(target)
    return run(target)


if __name__ == "__main__":
    raise SystemExit(main())
