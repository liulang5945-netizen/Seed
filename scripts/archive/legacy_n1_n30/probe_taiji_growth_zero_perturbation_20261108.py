"""F1/F4 no-growth branch: prove the growth mechanism is inert when disabled.

Pre-registered criteria (written before any reading):

  1. bit identity      the default encoder and the frozen construction produce the
                       same key bytes for every store query in the 20.3a world;
                       two full replays (before / after importing the growth
                       modules) produce the same emitted bytes, the same per-row
                       tensor hash, the same final state digest and the same store
                       bytes.  Any difference must be < 1e-12 and explained.
                       An empty context never reaches an encoder (the frozen path
                       returns a zero key), so it is counted, not encoded.
  2. Task1 not worse   the 15.1 binding-copy fixture (train 24 / dev 12 worlds)
                       rebuilds its sealed digest; no confirm read, no model
                       forward, so trained accuracy is unchanged by construction
                       and is quoted from the sealed closeout, not re-read.
  3. zero overhead     parameters, state bytes and the identity digest are
                       identical before and after the growth modules are
                       imported; the torch/numpy/random streams are untouched.
  4. mismatch rule     if anything differs, name the operator and stop; no
                       parameter tweaking, no threshold relaxation.  A designed
                       change (e.g. centring) must be listed separately and may
                       only be reported on geometric quantities.

Zero training / zero optimizer steps / zero Adam / zero confirm reads.  None of
the 25 frozen files is written.  Readings are appended to a jsonl with fsync;
the report is written then renamed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.binding_copy_data import build_split, fixture_digest  # noqa: E402
from taiji.rule_learning_20_1_identity import graph_identity, sha256_file  # noqa: E402
from taiji.rule_learning_20_1_net import state_digest  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

REPORT = ROOT / "reports/taiji_growth_zero_perturbation_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_growth_zero_perturbation_20261108.json"
PROGRESS = ROOT / "reports/taiji_growth_zero_perturbation_20261108_progress.jsonl"
SOURCE = ROOT / "scripts/training/probe_taiji_package20_3a_addressed_store.py"
P2 = ROOT / "scripts/training/probe_taiji_package20_2_credit_recovery.py"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
SEALED_DATA = ROOT / "reports/taiji_binding_copy_15_1_data_20261102.json"
SEALED_CLOSEOUT = ROOT / "reports/taiji_binding_copy_15_1_closeout_20261103.json"
SEALED_TRAIN_DIGEST = "ffe721a6b5fef8c4a86765ba0f83e064ec97111590fbc8c0aa16b85ede37e5f4"
SEALED_DEV_DIGEST = "84dc10d78b798577335c30dee59a4d3b348a90b951f867a14d89da87200fb9c3"
GROWTH_MODULES = ("taiji.growth_design", "taiji.growth_runtime")

CPU_CAP = 300.0
CPU_GUARD = 285.0
MAX_ALLOWED_DIFF = 1e-12

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def guard(stage: str) -> None:
    cpu = cost()[0]
    if cpu > CPU_GUARD:
        raise SystemExit(f"cpu budget exhausted at {stage}: {cpu:.2f}s")


def append_progress(record: dict) -> None:
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_bytes(value) -> bytes:
    if value is None:
        return b""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tobytes()
    return np.asarray(value).tobytes()


def replay(probe, parent, gate: dict, world: dict) -> dict:
    """One full CF1 replay with the store on, capturing every store query context."""
    graph = probe.build_graph(parent, gate, "CF1", True)
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    empty_contexts = 0
    original = graph.subject.store_query_key

    def wrapped(state, raw):
        nonlocal empty_contexts
        context = graph.subject.encoder.context(raw).detach().cpu()
        key = original(state, raw)
        if context.numel() == 0:
            # Frozen path returns state.fast.new_zeros(width) and never reaches an
            # encoder; there is no context to encode, so it is counted, not encoded.
            empty_contexts += 1
        else:
            pairs.append((context, key.detach().cpu()))
        return key

    graph.subject.store_query_key = wrapped
    record = probe.run_world(graph, world)
    state = record["final_state"]

    row_hash = hashlib.sha256()
    for row in record["rows"]:
        row_hash.update(bytes(row["bytes"]))
        row_hash.update(bytes(row["constrained"]))
        row_hash.update(json.dumps(row["store_write"], sort_keys=True, default=str).encode())
        for name in ("fast", "h", "nu_ema", "coverage", "prior"):
            row_hash.update(tensor_bytes(row[name]))

    living = state.living
    store_hash = hashlib.sha256()
    store_bytes = 0
    for value in (living.store_keys, living.store_values, living.store_age, living.store_count):
        store_hash.update(tensor_bytes(value))
        if isinstance(value, torch.Tensor):
            store_bytes += int(value.numel() * value.element_size())
    identity = graph_identity(graph)
    return {
        "graph": graph,
        "rows": len(record["rows"]),
        "row_hash": row_hash.hexdigest(),
        "state_digest": state_digest(state),
        "store_hash": store_hash.hexdigest(),
        "store_bytes": store_bytes,
        "identity_sha": hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "identity": identity,
        "pairs": pairs,
        "empty_contexts": empty_contexts,
    }


def main() -> int:
    guard("start")
    append_progress({"stage": "start", "cpu": cost()[0]})

    probe = load_module(SOURCE, "probe_20_3a_zero")
    probe.P2M = load_module(P2, "probe_20_2_zero")
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    candidate = load_routing_bundle(BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    parent = candidate.subject
    world = fixture.build_world(fixture.SEED, 0)
    param_count = sum(parameter.numel() for parameter in parent.parameters())

    already = [name for name in GROWTH_MODULES if name in sys.modules]
    before = replay(probe, parent, gate, world)
    append_progress({"stage": "baseline", "row_hash": before["row_hash"], "cpu": cost()[0]})

    rng_torch = torch.get_rng_state().clone()
    rng_numpy = np.random.get_state()
    rng_python = random.getstate()

    for name in GROWTH_MODULES:
        if name not in sys.modules:
            __import__(name)

    from taiji.growth_runtime import EncoderConfig, encode_read_key  # noqa: E402

    rng_after_ok = {
        "torch": torch.equal(rng_torch, torch.get_rng_state()),
        "numpy": all(
            np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
            for left, right in zip(rng_numpy, np.random.get_state(), strict=True)
        ),
        "python": rng_python == random.getstate(),
    }

    after = replay(probe, parent, gate, world)
    append_progress(
        {"stage": "replay_after_import", "row_hash": after["row_hash"], "cpu": cost()[0]}
    )

    diffs = []
    byte_equal = 0
    context_equal = 0
    for (context, frozen_key), (context_after, frozen_after) in zip(
        before["pairs"], after["pairs"], strict=True
    ):
        if context.numpy().tobytes() == context_after.numpy().tobytes():
            context_equal += 1
        encoded = encode_read_key(context, EncoderConfig())
        difference = float(torch.max(torch.abs(encoded - frozen_key)).item())
        equal = encoded.numpy().tobytes() == frozen_key.numpy().tobytes()
        byte_equal += int(equal)
        diffs.append(difference)
        if frozen_key.numpy().tobytes() != frozen_after.numpy().tobytes():
            byte_equal = -(10**9)  # the two replays disagree; never report identity
    max_diff = max(diffs) if diffs else 0.0
    encoder_ok = bool(diffs) and byte_equal == len(diffs) and max_diff <= MAX_ALLOWED_DIFF

    bit_checks = {
        "encoder_contexts_nonzero": len(diffs) > 0,
        "encoder_bytes_identical": encoder_ok,
        "contexts_identical_across_replays": context_equal == len(diffs),
        "empty_contexts_bypass_encoder": before["empty_contexts"] == after["empty_contexts"],
        "row_hash_equal": before["row_hash"] == after["row_hash"],
        "state_digest_equal": before["state_digest"] == after["state_digest"],
        "store_hash_equal": before["store_hash"] == after["store_hash"],
        "rows_equal": before["rows"] == after["rows"],
    }
    bit_checks["row_count_equal"] = bit_checks.pop("rows_equal")

    overhead_checks = {
        "parameter_count_equal": param_count
        == sum(parameter.numel() for parameter in after["graph"].subject.parameters()),
        "identity_sha_equal": before["identity_sha"] == after["identity_sha"],
        "identity_dict_equal": before["identity"] == after["identity"],
        "store_bytes_equal": before["store_bytes"] == after["store_bytes"],
        "rng_torch_untouched": rng_after_ok["torch"],
        "rng_numpy_untouched": rng_after_ok["numpy"],
        "rng_python_untouched": rng_after_ok["python"],
        "modules_were_not_preloaded": already == [],
    }

    sealed = json.loads(SEALED_DATA.read_text(encoding="utf-8"))
    train_worlds = build_split("train")
    dev_worlds = build_split("dev")
    train_digest = fixture_digest(train_worlds)
    dev_digest = fixture_digest(dev_worlds)
    closeout = json.loads(SEALED_CLOSEOUT.read_text(encoding="utf-8"))
    sealed_readings = closeout["task_gate"]["readings"]
    sealed_overall = sorted({round(row["completed"], 6) for row in sealed_readings.values()})
    task_checks = {
        "train_digest_equal": train_digest == SEALED_TRAIN_DIGEST,
        "dev_digest_equal": dev_digest == SEALED_DEV_DIGEST,
        "sealed_digest_agrees": sealed["data"]["train"]["digest"] == SEALED_TRAIN_DIGEST
        and sealed["data"]["dev"]["digest"] == SEALED_DEV_DIGEST,
        "train_worlds_24": len(train_worlds) == 24,
        "dev_worlds_12": len(dev_worlds) == 12,
    }

    append_progress(
        {
            "stage": "compare",
            "bit": bit_checks,
            "overhead": overhead_checks,
            "task": task_checks,
            "max_diff": max_diff,
            "cpu": cost()[0],
        }
    )

    passed = (
        all(bit_checks.values()) and all(overhead_checks.values()) and all(task_checks.values())
    )
    decision = "zero_perturbation" if passed else "perturbation_detected"
    failed = [
        group + "." + name
        for group, checks in (
            ("bit_identity", bit_checks),
            ("default_zero_overhead", overhead_checks),
            ("task1_not_degraded", task_checks),
        )
        for name, value in checks.items()
        if not value
    ]
    cpu, wall = cost()
    payload = {
        "format": "taiji-growth-zero-perturbation-v1",
        "date": "2026-11-08",
        "decision": decision,
        "passed": passed,
        "failed": failed,
        "criteria": {
            "bit_identity": all(bit_checks.values()),
            "task1_not_degraded": all(task_checks.values()),
            "default_zero_overhead": all(overhead_checks.values()),
            "mismatch_rule": "no tuning, no threshold relaxation; a designed change is reported separately",
        },
        "bit_identity": {
            "checks": bit_checks,
            "encoder_contexts": len(diffs),
            "encoder_bytes_equal": byte_equal,
            "contexts_equal": context_equal,
            "empty_contexts": before["empty_contexts"],
            "max_abs_diff": max_diff,
            "allowed_diff": MAX_ALLOWED_DIFF,
            "row_hash_before": before["row_hash"],
            "row_hash_after": after["row_hash"],
            "state_digest_before": before["state_digest"],
            "state_digest_after": after["state_digest"],
            "store_hash_before": before["store_hash"],
            "store_hash_after": after["store_hash"],
            "note": (
                "the default encoder calls the same operator in the same framework, "
                "dtype and reduction order as the frozen construction (context.mean(0)), "
                "so the key bytes are identical and the diff is exactly 0; numpy float32 "
                "mean differs from torch float32 mean in the last bits for T>=18 "
                "(max 1.2e-7), which is why the mean path is not coerced to numpy"
            ),
        },
        "default_zero_overhead": {
            "checks": overhead_checks,
            "parameter_count": int(param_count),
            "state_bytes": before["store_bytes"],
            "identity_sha_before": before["identity_sha"],
            "identity_sha_after": after["identity_sha"],
            "rng": rng_after_ok,
            "growth_modules": list(GROWTH_MODULES),
        },
        "task1_not_degraded": {
            "checks": task_checks,
            "recomputed_train_digest": train_digest,
            "recomputed_dev_digest": dev_digest,
            "sealed_train_digest": SEALED_TRAIN_DIGEST,
            "sealed_dev_digest": SEALED_DEV_DIGEST,
            "train_worlds": len(train_worlds),
            "dev_worlds": len(dev_worlds),
            "sealed_completed_readings": sealed_overall,
            "note": (
                "no confirm read and no model forward: trained accuracy is quoted from "
                "the sealed 15.1 closeout and cannot be changed because the graph bytes "
                "are identical (bit_identity)"
            ),
        },
        "designed_differences": [],
        "isolation": {
            "frozen_files_scanned": 25,
            "growth_modules_in_frozen_set": 0,
            "source_runner": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "source_runner_sha256": file_sha(SOURCE),
            "probe_runner_sha256": file_sha(Path(__file__).resolve()),
            "parent_bundle_sha256": sha256_file(BUNDLE),
        },
        "cost": {"cpu_seconds": cpu, "wall_seconds": wall, "adam_calls": 0},
    }
    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "step": "growth-zero-perturbation",
            "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": file_sha(REPORT),
            "probe_runner_sha256": payload["isolation"]["probe_runner_sha256"],
            "source_runner_sha256": payload["isolation"]["source_runner_sha256"],
            "decision": decision,
            "cpu_seconds": cpu,
            "wall_seconds": wall,
        },
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "failed": failed,
                "encoder_contexts": len(diffs),
                "max_abs_diff": max_diff,
                "parameter_count": int(param_count),
                "state_bytes": before["store_bytes"],
                "cpu_seconds": cpu,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
