"""Freeze the formal-check② B-case reference and parent lineage (zero training, 0 check).

Writes plans/manifests/taiji_u2_b_parent_reference_20261120.json with:
* D-reference text (D-53/54/55 conclusions) and sha256 pins of the evaluation/task
  code that check② must use unchanged;
* B-parent lineage: 22.5.1 candidate (fit_face_only, seed 20261112) reports and the
  8-path child subject capsules/reports (seed 0/1/2) with sha256 digests;
* explicit ``frozen`` markers and the three failure exits.

Zero training, zero check; re-running must reproduce identical digests.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/manifests/taiji_u2_b_parent_reference_20261120.json"

CODE_FILES = [
    "taiji/u2_neural_units.py",
    "taiji/u2_multihead.py",
    "taiji/u2_multihead2.py",
    "scripts/training/first_a_support_22_5_v2.py",
    "scripts/training/probe_s2_22_7_0_alpha_adapter_v3_20261120.py",
    "scripts/training/probe_s2_22_7_joint24_20261120.py",
    "scripts/training/probe_s2_action_consequence_20261120.py",
    "scripts/training/probe_s2_multihead2_20261120.py",
    "scripts/training/probe_s2_multihead5_20261120.py",
    "scripts/training/probe_s2_multihead7_20261120.py",
]

REPORTS = [
    "reports/taiji_u2_22_5_1_joint_gate_price_20261120.json",
    "reports/taiji_u2_22_5_1_identity_evidence_20261120.json",
    "reports/taiji_u2_22_7_0_alpha_adapter_v3_20261120.json",
    "reports/taiji_u2_22_4_2b_action_run_20261120.json",
    "reports/taiji_u2_multihead7_20261120.json",
    "reports/taiji_u2_multihead8_interference_20261120.json",
]

CAPSULES = [
    *(
        f"output/s2_multihead7/seed{seed}_session{session}.pt"
        for seed in (0, 1, 2)
        for session in (1, 8)
    ),
]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    missing = []
    entries: dict[str, dict[str, str]] = {"code": {}, "reports": {}, "capsules": {}}
    for group, paths in (("code", CODE_FILES), ("reports", REPORTS), ("capsules", CAPSULES)):
        for relative in paths:
            path = ROOT / relative
            if not path.exists():
                missing.append(relative)
                continue
            entries[group][relative] = sha256(path)
    manifest = {
        "schema": "taiji-u2-b-parent-reference-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_reference",
        "purpose": "formal check② case-B prerequisite: D-reference freeze + B-parent lineage",
        "check_reads": 0,
        "training_steps": 0,
        "d_reference": {
            "d53": "action card engineering_failure: fit rows mixed into train (fit leak)",
            "d54": "groups=4 routes only 0-31 dims; goal/feedback dead; use groups=6",
            "d55": "action_consequence_pass (bounded): N=3 push, position bins, four-arm counterfactual",
            "frozen_rule": "check② evaluation must reuse the pinned code and pre-registered arms; no post-hoc retuning",
        },
        "b_parent_lineage": {
            "parent": {
                "candidate": "22.5.1 joint-gate candidate",
                "task_reading": "fit_face_only (A case, working path)",
                "seed": 20261112,
                "reports": [
                    "reports/taiji_u2_22_5_1_joint_gate_price_20261120.json",
                    "reports/taiji_u2_22_5_1_identity_evidence_20261120.json",
                ],
            },
            "child": {
                "candidate": "8-pathway subject (joint24 + 7 per-task pathways)",
                "parents_per_seed": [20261120 + seed for seed in (0, 1, 2)],
                "capsules": [relative for relative in CAPSULES if (ROOT / relative).exists()],
                "reports": [
                    "reports/taiji_u2_multihead7_20261120.json",
                    "reports/taiji_u2_multihead8_interference_20261120.json",
                ],
                "evidence": "8 pathways retained drop 0.0; 8x8 interference others_max_delta=0.0",
            },
        },
        "failure_exits": ["unseen_task_failed", "unseen_task_inconclusive", "engineering_failure"],
        "budget": {"check2_cpu_cap": 540.0, "prerequisite_cpu": 40.0},
        "pins": entries,
        "missing": missing,
    }
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest["manifest_digest"] = digest
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST),
                "digest": digest,
                "pinned": {group: len(entries[group]) for group in entries},
                "missing": missing,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
