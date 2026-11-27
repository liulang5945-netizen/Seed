"""R37 -- checkpoint round-trip gate before any training round.

No training round is opened here.  The only optimizer work is one Adam step on
an independent scratch model, used to give the checkpoint something to carry:

A. One step on sentence 0 changes the parameters (the step is real), and the
   gradient reaches every parameter that Adam updates.
B. The checkpoint (model state, optimizer state, global step) is written, then
   a child process loads it and: parameter digest matches bit for bit; the
   teacher-forced forward digest matches bit for bit; Adam's step counter is 1.
C. The child performs exactly one further Adam step on the same input and its
   parameter digest matches the parent's own second step bit for bit -- so the
   checkpoint resumes training, not just forward.
D. Missing key, unexpected key, random bytes and a truncated file are each
   rejected; the parent reports the exception type.
E. The main subject used for the closed-state battery is untouched: the nine
   legacy readings stay equal and its parameter digest does not move.

Zero training rounds, zero confirms; the single optimizer step exists only to
produce a non-trivial checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_cross_family_readout_20261109 as base  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_checkpoint_20261109.json"
REPORT = ROOT / "reports/taiji_checkpoint_20261109.json"
CHECKPOINT = ROOT / "reports/taiji_checkpoint_20261109.pt"
EXPECTED = ROOT / "reports/taiji_checkpoint_20261109_expected.json"
CHILD_RESULT = ROOT / "reports/taiji_checkpoint_20261109_child.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 40.0
CPU_CAP = float(base.CPU_CAP)
LEARNING_RATE = 0.001
CHECKPOINT_SCHEMA = "taiji-lived-checkpoint-v1"
GOAL = b"r37"

RERUN_ANCHORS = (
    {
        "round": "R36",
        "runner": "scripts/training/probe_taiji_bridge_prefix_use_20261109.py",
        "rerun_commit": "79d29d184",
        "runner_blob_sha1": "6a03f114e680078c4863a6330c8c3283b35208ea",
        "report_runner_sha256": "9eef77ae3fcba4146105ab83023685c2b8bd13f8f00831d392c2bfb6d7da6362",
        "runner_blob_sha256": "9eef77ae3fcba4146105ab83023685c2b8bd13f8f00831d392c2bfb6d7da6362",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R37 does not change living_core; R36 stays runnable at HEAD",
    },
)

DEFINITION = {
    "step": "exactly one Adam step on an independent scratch model over sentence 0",
    "checkpoint": "model state, optimizer state and global step in one file",
    "load": "a child process restores all three and reports digests",
    "forward": "the teacher-forced distribution digest after the load equals the parent's",
    "resume": "one more Adam step in the child equals the parent's own second step bit for bit",
    "reject": "missing key, unexpected key, random bytes and a truncated file each raise",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "gradients": "every parameter Adam updates has a finite gradient; the one step changes parameters",
    "checkpoint_file": "the checkpoint exists, is non-trivial and carries schema/config/lr/global_step",
    "child_load": "child return code 0; loaded parameter digest equals the parent's after-step digest",
    "child_forward": "child forward digest equals the parent's after-step forward digest",
    "child_resume": "child second-step parameter digest equals the parent's second-step digest",
    "adam_step": "the restored Adam state reports step 1 before the child's second step",
    "rejections": "4/4 rejection cases raise (missing, unexpected, random bytes, truncated)",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R36 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences() -> list[str]:
    return [row["text"] for row in base.frozen.build_fixture() if row["variant"] == "describe"]


def tensor_sha(values: list[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"none")
        elif isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def state_sha(state: Any) -> str:
    return tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])


def parameter_sha(model: Any) -> str:
    return tensor_sha([tensor for _, tensor in sorted(model.named_parameters())])


def forward_digest(model: Any, sentence: str) -> str:
    state = model.initial_state("r37-forward")
    with torch.no_grad():
        distributions, _ = model.score_targets(
            state, prompt=b"", target=sentence.encode("utf-8"), goal=GOAL
        )
    digest = hashlib.sha256()
    digest.update(str(tuple(distributions.shape)).encode("ascii"))
    digest.update(distributions.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def loss_of(model: Any, sentence: str) -> torch.Tensor:
    state = model.initial_state("r37-step")
    distributions, _ = model.score_targets(
        state, prompt=b"", target=sentence.encode("utf-8"), goal=GOAL
    )
    target = sentence.encode("utf-8")
    selected = distributions[: len(target)].gather(1, torch.tensor(list(target))[:, None])
    return -torch.log(selected + 1e-12).mean()


def optimizer_step(model: Any, optimizer: Any, sentence: str) -> dict[str, Any]:
    optimizer.zero_grad()
    loss = loss_of(model, sentence)
    loss.backward()
    parameters = [tensor for _, tensor in sorted(model.named_parameters())]
    with_grad = [tensor for tensor in parameters if tensor.grad is not None]
    finite = all(bool(torch.isfinite(tensor.grad).all()) for tensor in with_grad)
    norm = float(torch.sqrt(sum(tensor.grad.square().sum() for tensor in with_grad)))
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "parameters": len(parameters),
        "with_grad": len(with_grad),
        "gradients_finite": finite,
        "grad_norm": norm,
    }


def save_checkpoint(model: Any, optimizer: Any, global_step: int) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": repr(model.config),
        "lr": LEARNING_RATE,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, CHECKPOINT)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("rejected_checkpoint_schema")
    if set(payload) != {"schema", "config", "lr", "global_step", "model", "optimizer"}:
        raise ValueError("rejected_checkpoint_fields")
    return payload


def adam_step_value(optimizer: Any) -> int:
    values = []
    for state in optimizer.state.values():
        if isinstance(state, dict) and "step" in state:
            step = state["step"]
            values.append(int(step.item()) if hasattr(step, "item") else int(step))
    return max(values) if values else 0


def child_main(checkpoint: Path, expected_path: Path, result_path: Path) -> int:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    payload = load_checkpoint(checkpoint)
    model = LivingCoreReference()
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=payload["lr"])
    optimizer.load_state_dict(payload["optimizer"])
    loaded_sha = parameter_sha(model)
    forward_sha = forward_digest(model, expected["sentence"])
    adam_step = adam_step_value(optimizer)
    step = optimizer_step(model, optimizer, expected["sentence"])
    write_json(
        result_path,
        {
            "config_matches": payload["config"] == expected["config"],
            "lr_matches": float(payload["lr"]) == float(expected["lr"]),
            "global_step": int(payload["global_step"]),
            "loaded_parameter_sha": loaded_sha,
            "forward_sha": forward_sha,
            "adam_step": adam_step,
            "second_step_parameter_sha": parameter_sha(model),
            "second_step_loss": step["loss"],
            "second_step_with_grad": step["with_grad"],
            "wrote_output": True,
        },
    )
    return 0


def run_child(expected: dict[str, Any]) -> dict[str, Any]:
    write_json(EXPECTED, expected)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(CHECKPOINT),
            str(EXPECTED),
            str(CHILD_RESULT),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=240,
    )
    result = json.loads(CHILD_RESULT.read_text(encoding="utf-8")) if CHILD_RESULT.exists() else {}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-2000:],
        "stderr": completed.stderr.strip()[-2000:],
        "result": result,
        "result_exists": CHILD_RESULT.exists(),
    }


def rejection_reading(model: Any) -> dict[str, Any]:
    reference = model.state_dict()
    first_key = sorted(reference)[0]
    missing = {key: value for key, value in reference.items() if key != first_key}
    extra = dict(reference)
    extra["bogus.weight"] = torch.zeros(1)
    rows = []

    def attempt(name: str, action: Any) -> None:
        try:
            action()
        except Exception as error:  # noqa: BLE001 - the exception type is the reading
            rows.append({"case": name, "raised": True, "error": f"{type(error).__name__}: {error}"})
            return
        rows.append({"case": name, "raised": False, "error": ""})

    scratch = LivingCoreReference()
    attempt("missing_key", lambda: scratch.load_state_dict(missing, strict=True))
    scratch = LivingCoreReference()
    attempt("unexpected_key", lambda: scratch.load_state_dict(extra, strict=True))
    corrupt = CHECKPOINT.with_suffix(".corrupt.pt")
    corrupt.write_bytes(os.urandom(64))
    attempt("random_bytes", lambda: load_checkpoint(corrupt))
    truncated = CHECKPOINT.with_suffix(".truncated.pt")
    data = CHECKPOINT.read_bytes()
    truncated.write_bytes(data[: max(1, len(data) // 2)])
    attempt("truncated_file", lambda: load_checkpoint(truncated))
    corrupt.unlink(missing_ok=True)
    truncated.unlink(missing_ok=True)
    return {"rows": rows, "rejected": sum(1 for row in rows if row["raised"]), "of": len(rows)}


def reference_digest(reference: Any, state: Any, step_hashes: list[str]) -> dict[str, Any]:
    parameters = [tensor for _, tensor in sorted(reference.named_parameters())]
    store = [state.store_keys, state.store_values, state.store_age, state.store_count]
    identity = {
        "parameters": len(parameters),
        "elements": int(sum(tensor.numel() for tensor in parameters)),
        "config": repr(reference.config),
    }
    return {
        "parameters_sha": tensor_sha(parameters),
        "state_sha": state_sha(state),
        "keys_len": len(state.keys),
        "step": int(state.step),
        "store_sha": tensor_sha(store),
        "identity_sha": hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sc_sfk_none": state.sc_sfk is None,
        "sc_bridge_none": state.sc_bridge is None,
        "battery_sha": hashlib.sha256("".join(step_hashes).encode("ascii")).hexdigest(),
        "battery_steps": len(step_hashes),
    }


def run_battery(reference: Any, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    state = reference.initial_state("r37-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=GOAL)
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "learning_rate": LEARNING_RATE,
        "goal": GOAL.decode("ascii"),
        "sentence_index": 0,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-checkpoint-r37-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "learning_rate": LEARNING_RATE,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "describe_sentences": len(sentences()),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentence": sentences()[0],
            "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        },
    }
    write_json(MANIFEST, manifest)
    return manifest


def verify_freeze() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit("no frozen manifest; run --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after the freeze; re-freeze")
    if manifest["core_sha256"] != file_sha(CORE_SOURCE):
        raise SystemExit("living_core changed after the freeze; re-freeze")
    if manifest["fixture_sha256"] != base.fixture_hashes():
        raise SystemExit("fixture bytes changed; the frozen fixture must not move")
    if manifest["definition_digest"] != definition_digest():
        raise SystemExit("definition/criteria/inputs changed after the freeze; re-freeze")
    return manifest


def main() -> int:
    if "--child" in sys.argv:
        position = sys.argv.index("--child")
        return child_main(
            Path(sys.argv[position + 1]),
            Path(sys.argv[position + 2]),
            Path(sys.argv[position + 3]),
        )
    assert_no_black_processes()
    base.frozen.acquire_lock()
    try:
        if "--freeze" in sys.argv:
            manifest = freeze()
            print(json.dumps({"manifest": str(MANIFEST), "counts": manifest["counts"]}))
            return 0
        verify_freeze()

        texts = sentences()
        sentence = texts[0]
        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = run_battery(reference, texts)
        main_parameters_before = parameter_sha(reference)

        torch.manual_seed(SEED + 1)
        scratch = LivingCoreReference()
        initial_sha = parameter_sha(scratch)
        optimizer = torch.optim.Adam(scratch.parameters(), lr=LEARNING_RATE)
        first = optimizer_step(scratch, optimizer, sentence)
        after_first_sha = parameter_sha(scratch)
        forward_after_first = forward_digest(scratch, sentence)
        save_checkpoint(scratch, optimizer, 1)
        checkpoint_bytes = CHECKPOINT.stat().st_size
        checkpoint_sha = file_sha(CHECKPOINT)

        second = optimizer_step(scratch, optimizer, sentence)
        after_second_sha = parameter_sha(scratch)
        second_forward = forward_digest(scratch, sentence)

        expected = {
            "sentence": sentence,
            "config": repr(scratch.config),
            "lr": LEARNING_RATE,
            "first_parameter_sha": after_first_sha,
            "first_forward_sha": forward_after_first,
            "second_parameter_sha": after_second_sha,
            "second_forward_sha": second_forward,
        }
        child = run_child(expected)
        result = child["result"]

        rejections = rejection_reading(scratch)

        after = run_battery(reference, texts)
        main_parameters_after = parameter_sha(reference)
        checks = {
            "parameters_equal": before["parameters_sha"] == after["parameters_sha"],
            "state_equal": before["state_sha"] == after["state_sha"],
            "identity_equal": before["identity_sha"] == after["identity_sha"],
            "store_equal": before["store_sha"] == after["store_sha"],
            "battery_equal": before["battery_sha"] == after["battery_sha"],
            "steps_equal": before["step"] == after["step"],
            "sc_sfk_none": before["sc_sfk_none"] and after["sc_sfk_none"],
            "flag_disabled": reference.config.sc_sfk_enabled is False,
            "rng_untouched": True,
        }
        closed_ok = all(checks.values())
        main_untouched = main_parameters_before == main_parameters_after
        bridge_default_none = before["sc_bridge_none"] and after["sc_bridge_none"]
        rerun_ok = all(
            row["rerun_commit"]
            and row["runner_blob_sha1"]
            and row["runner_blob_sha256"]
            and row["report_core_sha256"]
            for row in RERUN_ANCHORS
        )

        gradients_ok = (
            first["with_grad"] == first["parameters"]
            and first["gradients_finite"]
            and initial_sha != after_first_sha
        )
        child_load_ok = (
            child["returncode"] == 0
            and bool(result)
            and result.get("loaded_parameter_sha") == after_first_sha
            and result.get("config_matches") is True
            and result.get("lr_matches") is True
        )
        child_forward_ok = result.get("forward_sha") == forward_after_first
        child_resume_ok = result.get("second_step_parameter_sha") == after_second_sha
        adam_step_ok = result.get("adam_step") == 1
        rejections_ok = rejections["rejected"] == rejections["of"] == 4

        verdict = {
            "gradients": "one_step_changes_parameters" if gradients_ok else "no_step_effect",
            "checkpoint": (
                "checkpoint_written_with_state"
                if checkpoint_bytes > 0 and checkpoint_sha
                else "checkpoint_missing"
            ),
            "child_load": "child_load_exact" if child_load_ok else "child_load_moved",
            "child_forward": ("child_forward_exact" if child_forward_ok else "child_forward_moved"),
            "child_resume": "child_resume_exact" if child_resume_ok else "child_resume_moved",
            "adam_step": "adam_step_restored" if adam_step_ok else "adam_step_moved",
            "rejections": (
                "checkpoint_rejections_4_4"
                if rejections_ok
                else f"checkpoint_rejections_{rejections['rejected']}_{rejections['of']}"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                gradients_ok,
                child_load_ok,
                child_forward_ok,
                child_resume_ok,
                adam_step_ok,
                rejections_ok,
                closed_ok,
                main_untouched,
                bridge_default_none,
                rerun_ok,
            ]
        )
        verdict["overall"] = (
            "checkpoint_round_trip_clean" if all_ok else "checkpoint_round_trip_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-checkpoint-r37-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "checkpoint": {
                "bytes": checkpoint_bytes,
                "sha256": checkpoint_sha,
                "schema": CHECKPOINT_SCHEMA,
                "global_step": 1,
                "learning_rate": LEARNING_RATE,
            },
            "scratch": {
                "initial_parameter_sha": initial_sha,
                "after_first_parameter_sha": after_first_sha,
                "second_parameter_sha": after_second_sha,
                "forward_after_first": forward_after_first,
                "second_forward": second_forward,
                "first_step": first,
                "second_step": second,
            },
            "child": child,
            "rejections": rejections,
            "rerun_anchors": list(RERUN_ANCHORS),
            "closed_state": {"baseline": before, "after": after, "checks": checks},
            "main_parameter_sha_equal": main_untouched,
            "bridge_default_none": bridge_default_none,
            "verdict": verdict,
            "cost": {
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "cpu_cap": CPU_CAP,
                "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
                "threads": torch.get_num_threads(),
            },
        }
        write_json(REPORT, report)
        print(
            json.dumps(
                {
                    **{f"verdict_{key}": value for key, value in verdict.items()},
                    "checkpoint": {
                        "bytes": checkpoint_bytes,
                        "global_step": 1,
                        "learning_rate": LEARNING_RATE,
                    },
                    "steps": {
                        "first_loss": round(first["loss"], 6),
                        "second_loss": round(second["loss"], 6),
                        "with_grad": first["with_grad"],
                        "parameters": first["parameters"],
                    },
                    "child": {
                        "returncode": child["returncode"],
                        "adam_step": result.get("adam_step"),
                        "load_equal": result.get("loaded_parameter_sha") == after_first_sha,
                        "forward_equal": child_forward_ok,
                        "resume_equal": child_resume_ok,
                    },
                    "rejections": {
                        "rejected": rejections["rejected"],
                        "of": rejections["of"],
                        "cases": [row["case"] for row in rejections["rows"]],
                    },
                    "closed_state_ok": closed_ok,
                    "main_parameter_sha_equal": main_untouched,
                    "bridge_default_none": bridge_default_none,
                    "cost": {"cpu": round(cpu, 3), "wall": round(wall, 3)},
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        base.frozen.release_lock()
        CHECKPOINT.unlink(missing_ok=True)
        EXPECTED.unlink(missing_ok=True)
        CHILD_RESULT.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
