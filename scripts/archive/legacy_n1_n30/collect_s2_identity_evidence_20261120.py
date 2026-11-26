"""Zero-training identity / lineage evidence collection for the 22.5.1 child bundle.

Reads the accepted v1 parent bundle and the three saved joint-gate children and
records: bundle digests, inherited-parameter bit-equality, organ key/parameter
inventory, current organ mixing, and the cross-process recovery equality already
produced by the joint-gate run.  Zero optimizer steps, zero check/final reads,
no capability signing: this is machine-readable lineage evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.s0_parent_loader_20261119 import (  # noqa: E402
    CHECKPOINTS,
    load_historical,
)
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_1_identity_evidence_20261120.json"
ARTIFACTS = ROOT / "output/s2_joint_gate_20261120"
RUN = ROOT / "reports/taiji_u2_22_5_1_joint_gate_run_20261120.json"
PARENT_SEED = 20261112
SEEDS = (20261110, 20261111, 20261112)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    parent, _payload, audit = load_historical(CHECKPOINTS[PARENT_SEED])
    parent_state = parent.state_dict()
    parent_digest = value_digest(parent_state)
    run = json.loads(RUN.read_text(encoding="utf-8"))
    seeds_rows = {row["seed"]: row for row in run["seeds"]}
    verify = json.loads(
        (ROOT / "reports/taiji_u2_22_5_1_joint_gate_verify_20261120.json").read_text(
            encoding="utf-8"
        )
    )
    amended_a = {entry["seed"]: entry["a_child"] for entry in verify.get("per_seed", [])}
    children = []
    for seed in SEEDS:
        artifact = ARTIFACTS / f"child-{seed}.pt"
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        child = U2Subject(U2Config(**payload["config"]))
        child.load_state_dict(payload["state_dict"])
        child_state = child.state_dict()
        inherited = [name for name in parent_state if not name.startswith("record_organ.")]
        organ_keys = [name for name in child_state if name.startswith("record_organ.")]
        max_inherited_diff = max(
            float((parent_state[name] - child_state[name]).abs().max()) for name in inherited
        )
        inherited_equal = all(
            torch.equal(parent_state[name], child_state[name]) for name in inherited
        )
        organ_params = sum(
            parameter.numel()
            for name, parameter in child.named_parameters()
            if name.startswith("record_organ.")
        )
        recovery = seeds_rows[seed]["recovery_probe"]["readings"]
        recover_receipt = json.loads(
            (ARTIFACTS / f"recover-{seed}.json").read_text(encoding="utf-8")
        )
        children.append(
            {
                "training_seed": seed,
                "artifact": str(artifact.relative_to(ROOT)),
                "artifact_sha256": sha256_of(artifact),
                "state_digest": value_digest(child_state),
                "config": {
                    "readout": child.c.readout,
                    "d": child.c.d,
                    "record_organ": child.c.record_organ,
                },
                "inherited_keys": len(inherited),
                "inherited_bit_identical": inherited_equal,
                "inherited_max_abs_diff": max_inherited_diff,
                "organ_keys": len(organ_keys),
                "organ_keys_sample": organ_keys[:4],
                "organ_parameters": organ_params,
                "organ_mix": float(child.record_organ.organ_mix.detach()),
                "recovery_readings_sha256": hashlib.sha256(
                    json.dumps(recovery, sort_keys=True).encode()
                ).hexdigest(),
                "recovery_cross_process_exact": recover_receipt["readings"] == recovery,
                "fit_face_readings": {
                    "b_correct": seeds_rows[seed]["arms"]["main"]["b_correct"],
                    "d_correct": seeds_rows[seed]["arms"]["main"]["d_correct"],
                    "a_correct": amended_a.get(
                        seed, str(seeds_rows[seed]["arms"]["main"]["a_correct"])
                    ),
                },
            }
        )
    payload = {
        "schema": "taiji-u2-22-5-1-identity-evidence-v1",
        "card": "22.5.1 joint-gate child lineage evidence (fit face)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "parent": {
            "seed": PARENT_SEED,
            "bundle": str(CHECKPOINTS[PARENT_SEED].relative_to(ROOT)),
            "bundle_sha256": sha256_of(CHECKPOINTS[PARENT_SEED]),
            "state_digest": parent_digest,
            "identity": audit.get("identity"),
            "strict_load": audit.get("strict_load"),
            "params": sum(p.numel() for p in parent.parameters()),
        },
        "lineage": {
            "rule": "single main candidate parent x three training child seeds (no parent pooling)",
            "parent_seed": PARENT_SEED,
            "children": children,
        },
        "claims": {
            "inheritance": "all non-organ inherited parameters are bit-identical to the accepted parent",
            "plasticity": "each child adds exactly the record-organ parameter set; no other keys",
            "recovery": "save -> independent process reload -> same-condition readings byte-identical",
            "scope": "fit face only; not a check adjudication; no capability signing",
        },
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "parent_digest": parent_digest[:16],
                "children": [
                    {
                        "seed": child["training_seed"],
                        "inherited_bit_identical": child["inherited_bit_identical"],
                        "organ_parameters": child["organ_parameters"],
                        "recovery_exact": child["recovery_cross_process_exact"],
                    }
                    for child in children
                ],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
