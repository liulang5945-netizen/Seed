"""22.8 S1: candidate mapping audit, segment-index freeze and zero-training smoke.

Key question frozen here: can the 22.5.1 B parent bundle (`child-20261112.pt`, native
core with d=32/groups=2/units=12/pixel-salience readout) be the same subject for the
22.x task faces (joint24, action, C task) computed on the 30-unit RouteBGraph?  There is
no parameter mapping between the two schemas, so this card records
``candidate_mapping_missing`` and freezes the segment index for the owner decision.
Zero training, zero check/final.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from scripts.training.probe_s2_action_consequence_20261120 import build_partition  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec, build_capsule  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_8_s1_20261120.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_22_8_segment_index_20261120.json"
PARENT = ROOT / "output/s2_joint_gate_20261120/child-20261112.pt"
CAP_CPU = 250.0
SESSIONS = 3
ROUNDS = 32
ALLOCATION = {
    "appearance": 8,
    "dynamics": 8,
    "occlusion_conflict": 6,
    "correction": 4,
    "no_prompt": 6,
}
MASTER_SEED = 20261120


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segment_index() -> dict[str, Any]:
    rounds = []
    for session in range(1, SESSIONS + 1):
        labels = []
        for name, count in ALLOCATION.items():
            labels.extend([name] * count)
        generator = torch.Generator().manual_seed(MASTER_SEED + session)
        order = torch.randperm(ROUNDS, generator=generator).tolist()
        for position in order:
            rounds.append(
                {
                    "session": session,
                    "round": len(rounds) - (session - 1) * ROUNDS + 1,
                    "type": labels[position],
                    "world_seed": MASTER_SEED + session * 1000 + position,
                }
            )
    payload = json.dumps(rounds, sort_keys=True, ensure_ascii=False).encode()
    return {"rounds": rounds, "rounds_sha256": hashlib.sha256(payload).hexdigest()}


def mapping_audit() -> dict[str, Any]:
    obj = torch.load(PARENT, map_location="cpu", weights_only=False)
    native = obj["state_dict"]
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=24, seed=20261120
    )
    route = RouteBGraph(spec)
    route_names = set(route.state_dict().keys())
    native_names = set(native.keys())
    shared = route_names & native_names
    return {
        "parent_path": str(PARENT.relative_to(ROOT)),
        "parent_sha256": sha_of(PARENT),
        "parent_config": obj["config"],
        "native_tensor_count": len(native_names),
        "route_tensor_count": len(route_names),
        "shared_names": sorted(shared),
        "shared_shapes": {
            name: (tuple(native[name].shape), tuple(route.state_dict()[name].shape))
            for name in shared
        },
        "mapping": "absent" if not shared else "partial_name_collision_only",
        "native_units": obj["config"].get("units"),
        "route_units": spec.n_units,
    }


def smoke() -> dict[str, Any]:
    examples = generate_examples("fit", 32)
    x_appearance = torch.stack([adapter_v3_features(item) for item in examples]).float()
    y_appearance = torch.tensor([joint_label(item) for item in examples])
    action_rows = build_partition("train", 32, 20261120)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=24, seed=20261120
    )
    graph = RouteBGraph(spec)
    capsule = build_capsule(graph)
    return {
        "appearance_shape": list(x_appearance.shape),
        "appearance_labels": sorted(set(y_appearance.tolist()))[:5],
        "dynamics_rows": len(action_rows),
        "dynamics_feature_dim": int(action_rows[0]["features"].numel()),
        "capsule_schema": capsule.get("schema"),
        "capsule_digest": capsule.get("capsule_digest", "")[:16],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    index = segment_index()
    audit = mapping_audit()
    smoke_result = smoke()
    gates = {
        "fixture_frozen": len(index["rounds"]) == SESSIONS * ROUNDS,
        "mapping_present": audit["mapping"] != "absent",
    }
    verdict = "candidate_mapping_missing" if not gates["mapping_present"] else "s1_ready"
    if not gates["fixture_frozen"]:
        verdict = "fixture_invalid"
    payload = {
        "schema": "taiji-u2-22-8-s1-v1",
        "card": "22.8 S1 candidate mapping audit + segment index freeze",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "sessions": SESSIONS,
            "rounds": ROUNDS,
            "allocation": ALLOCATION,
            "master_seed": MASTER_SEED,
            "check_reads": 0,
            "final_reads": 0,
        },
        "segment_index": {
            "rounds": len(index["rounds"]),
            "rounds_sha256": index["rounds_sha256"],
            "first_rounds": index["rounds"][:4],
        },
        "mapping_audit": audit,
        "smoke": smoke_result,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0, "optimizer_steps": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    contract = {
        "schema": "taiji-u2-22-8-segment-index-v1",
        "card": "22.8 S1 frozen segment index (3 sessions x 32 rounds)",
        "status": "frozen" if verdict == "s1_ready" else "frozen_blocked",
        "criteria": {
            "sessions": SESSIONS,
            "rounds": ROUNDS,
            "allocation": ALLOCATION,
            "master_seed": MASTER_SEED,
            "rounds_sha256": index["rounds_sha256"],
            "gate": "J1-J6 per DESIGN-TAIJI_U2_22_8_JOINT_LIFE_20261120.md",
        },
        "blocker": None if verdict == "s1_ready" else audit["mapping"],
        "check_reads": 0,
        "final_reads": 0,
    }
    contract["criteria_sha256"] = hashlib.sha256(
        json.dumps(contract["criteria"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    CONTRACT.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "shared_names": audit["shared_names"],
                "native_tensors": audit["native_tensor_count"],
                "route_tensors": audit["route_tensor_count"],
                "rounds": payload["segment_index"]["rounds"],
                "rounds_sha256": payload["segment_index"]["rounds_sha256"],
                "smoke": smoke_result,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "s1_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
