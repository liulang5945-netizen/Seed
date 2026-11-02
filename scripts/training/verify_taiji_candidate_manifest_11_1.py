"""11.1: manifest-bound isolated loading for the current candidate.

Builds a manifest for the 09.8 bundle, loads it in this process and again in a fresh
child process, and checks three refusals:

* the two loads agree on the parameter digest;
* a manifest with a tampered module hash is refused;
* a bundle whose bytes changed is refused.

Nothing is trained; the bundle is opened read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.candidate_manifest import (  # noqa: E402
    build_manifest,
    check_manifest,
    load_candidate,
    parameter_digest,
)

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600

CHILD = """
import json, sys
from pathlib import Path
sys.path.insert(0, {root!r})
from taiji.candidate_manifest import load_candidate, parameter_digest
manifest, payload = load_candidate(
    Path({root!r}), Path(sys.argv[1]), Path(sys.argv[2]), "frozen_eval"
)
print(json.dumps({{"digest": parameter_digest(payload["parameters"])}}))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.1 card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    bundle = ROOT / BUNDLE
    manifest = build_manifest(ROOT, bundle)

    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        loaded_manifest, payload = load_candidate(ROOT, manifest_path, bundle, "frozen_eval")
        local_digest = parameter_digest(payload["parameters"])
        identity_ok = local_digest == manifest["candidate_id"]

        child = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD.format(root=str(ROOT)),
                str(manifest_path),
                str(bundle),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        child_digest = None
        if child.returncode == 0:
            child_digest = json.loads(child.stdout.strip().splitlines()[-1])["digest"]
        child_ok = child_digest == local_digest

        tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_module = next(iter(tampered["modules"]))
        tampered["modules"][first_module] = "0" * 64
        tampered_path = Path(directory) / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8", newline="\n")
        module_refused = bool(check_manifest(ROOT, tampered, bundle))
        try:
            load_candidate(ROOT, tampered_path, bundle, "frozen_eval")
            module_raised = False
        except ValueError:
            module_raised = True

        copied = Path(directory) / "bundle_copy.pt"
        copied.write_bytes(bundle.read_bytes() + b"\x00")
        bytes_refused = bool(check_manifest(ROOT, manifest, copied))
        try:
            load_candidate(ROOT, manifest_path, copied, "frozen_eval")
            bytes_raised = False
        except ValueError:
            bytes_raised = True

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif not (identity_ok and child_ok):
        status = "identity_mismatch"
    elif not (module_refused and module_raised):
        status = "module_drift_accepted"
    elif not (bytes_refused and bytes_raised):
        status = "bundle_drift_accepted"
    else:
        status = "isolated_load_verified"

    report = {
        "schema": "taiji-candidate-manifest-11-1-v1",
        "step": "11.1",
        "date": "2026-11-01",
        "status": status,
        "candidate_id": manifest["candidate_id"],
        "subject_id": manifest["subject_id"],
        "revision": manifest["revision"],
        "step_in_bundle": manifest["step"],
        "config_fields": len(manifest["config"]),
        "modules_bound": len(manifest["modules"]),
        "parameters": manifest["state_schema"]["parameters"],
        "bundle": manifest["provenance"],
        "modes": list(manifest["modes"]),
        "identity_in_process": identity_ok,
        "identity_in_child": child_ok,
        "child_returncode": child.returncode,
        "module_drift_refused": module_refused and module_raised,
        "bundle_drift_refused": bytes_refused and bytes_raised,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, indent=2).encode()
        ).hexdigest(),
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Isolation and provenance only: a fresh process rebuilt the same identity from "
            "the manifest and refused two kinds of drift. It says nothing about behaviour, "
            "does not test task capability, and opens the bundle read-only."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/candidate_manifest.py",
                "scripts/training/verify_taiji_candidate_manifest_11_1.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "identity_in_process": identity_ok,
                "identity_in_child": child_ok,
                "module_drift_refused": module_refused and module_raised,
                "bundle_drift_refused": bytes_refused and bytes_raised,
                "candidate_id": manifest["candidate_id"][:16],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
