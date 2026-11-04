"""Build the 18.0 data premise (option A, non-training part).

Writes the expanded corpus, its manifest with the disjointness proofs, and the
pre-registration package.  It performs no training, no optimiser step, and it
does not generate confirm examples.

Run: ``python scripts/training/build_taiji_data_premise_18_0.py``
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji import data_premise_18_0 as premise  # noqa: E402

MODULE_PATH = ROOT / "taiji/data_premise_18_0.py"
OUTPUT = ROOT / "output/taiji_data_premise_18_0/data.json"
MANIFEST = ROOT / "plans/manifests/taiji_data_premise_18_0_manifest_20261104.json"
PREREG = ROOT / "plans/manifests/taiji_data_premise_18_0_prereg_20261104.json"

PARAMETER_COUNT = 3563859
TRAINABLE_TENSORS = 77
CANDIDATE_SHA_V2 = "773dcec2"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capacity_note(stats: dict) -> dict:
    """Names are record bytes, so the frozen graph does not grow."""
    return {
        "container_parameters": PARAMETER_COUNT,
        "trainable_tensors": TRAINABLE_TENSORS,
        "candidate_sha_v2": CANDIDATE_SHA_V2,
        "parameter_delta": 0,
        "reason": "names live in source bytes, not embeddings; the 27-byte action and 36-byte "
        "goal are unchanged",
        "optional_scales": {
            "default_220x220": {
                "object_names": 220,
                "place_names": 220,
                "train_examples": stats["train_examples"],
                "dev_examples": stats["dev_examples"],
                "parameter_delta": 0,
            },
            "wide_440x440": {
                "object_names": 440,
                "place_names": 440,
                "train_examples": stats["train_examples"] * 2,
                "dev_examples": stats["dev_examples"] * 2,
                "parameter_delta": 0,
                "note": "only needed if held-out accuracy saturates with the default pool",
            },
        },
    }


def prereg(payload: dict, stats: dict, proofs: dict, data_sha: str) -> dict:
    dev = premise.fixtures(payload, "dev")
    return {
        "schema": "taiji-data-premise-18-0-prereg-v1",
        "step": "N18.0",
        "date": "2026-11-04",
        "status": "awaiting_owner_approval",
        "binding": "frozen after owner approval; no training may start before that",
        "data": {
            "file_sha256": data_sha,
            "digest": payload["digest"],
            "train_examples": stats["train_examples"],
            "dev_examples": stats["dev_examples"],
            "confirm_examples_sealed": stats["confirm_examples"],
        },
        "gates": {
            "overall": 0.80,
            "unseen_output_name": 0.80,
            "new_description": 0.80,
            "reversed": 0.95,
            "selection_object": 0.90,
            "selection_place": 0.90,
            "content_drop": 0.10,
            "identity_rescue": 8,
            "a_minus_c": 0.10,
            "new_name_copied": 0.60,
        },
        "definitions": {
            "overall": "dev completed rate: full 27-byte action plus EOS, denominator all dev "
            "examples",
            "unseen_output_name": "completed rate on the unseen_output_name subset; denominator is "
            "the subset length",
            "new_description": "completed rate on the new_description subset; denominator is the "
            "subset length",
            "reversed": "completed rate when the record order is reversed",
            "selection_object": "top-1 accuracy of the object record",
            "selection_place": "top-1 accuracy of the place record",
            "content_drop": "completed(with records) - completed(no records), on the "
            "content-intervention subset; records must be necessary, not decorative",
            "new_name_copied": "fraction of dev examples where the object name AND the place name "
            "appear byte-exact in the output with records intact; this is an independent gate and "
            "must not be replaced by 'the output changed'",
            "identity_rescue": "count of dev probes whose probabilities after a forced-route no-op "
            "round trip equal the baseline bitwise, denominator unchanged from 17.3",
            "a_minus_c": "route-lock minus unlocked overall delta; two-seed world bootstrap lower "
            "bound must exceed zero",
        },
        "reporting": {
            "gamma": "every cell reports gamma_mean and 1-gamma_mean side by side; a missing side is "
            "an invalid row",
            "new_name_copied_floor": "a cell with overall >= 0.80 but new_name_copied < 0.60 is "
            "labelled recitation_remains, not a capability gain",
        },
        "axes": {
            "dev_examples": len(dev),
            "dev_unique_names": proofs["name_counts"]["dev"]["object_unique"]
            + proofs["name_counts"]["dev"]["place_unique"],
            "dev_examples_unseen_output_name": sum(
                1 for fixture in dev if fixture["flags"]["unseen_output_name"]
            ),
            "dev_examples_new_description": sum(
                1 for fixture in dev if fixture["flags"]["new_description"]
            ),
            "dev_examples_both_new": sum(1 for fixture in dev if fixture["flags"]["both_new"]),
            "dev_combinations_seen_in_train": proofs["combination_intersection_train_dev"],
            "train_names_in_dev_texts": proofs["occurrences"].get("train_object_names_in_dev", 0)
            + proofs["occurrences"].get("train_place_names_in_dev", 0),
        },
        "ab_relation": {
            "A": "this premise: large disjoint name pools plus combination held-out; run first",
            "B": "gamma forced to 0 at content slots; interpretable only after A because without "
            "held-out names its gain cannot be separated from recitation",
            "C": "rejected: A and B together change two variables and break attribution",
        },
        "budget_reference": {
            "manifest": "plans/manifests/taiji_data_premise_18_0_pricing_20261104.json",
            "note": "bare price plus the 2.11 external-contention protection; the manifest is the "
            "single source for the numbers, this file only points at it",
        },
        "stop_conditions": [
            "owner approval is required before any training",
            "task gate failure at step300 is a negative result; no seed or dose top-up",
            "step150 readiness failure stops the cell before step300",
            "no confirm read before the candidate and evaluator are frozen",
        ],
    }


def main() -> None:
    validation = premise.validate_pools()
    payload = premise.build_payload()
    proofs = premise.split_proofs(payload)
    stats = premise.stats(payload)
    payload["digest"] = premise.digest(payload)

    if proofs["nonzero_occurrence_keys"]:
        raise RuntimeError(f"name leakage: {proofs['nonzero_occurrence_keys']}")
    if proofs["combination_intersection_train_dev"]:
        raise RuntimeError("train/dev combination intersection is not empty")
    expected_zero = {
        "dev_object_names_in_train",
        "dev_place_names_in_train",
        "confirm_object_names_in_train",
        "confirm_place_names_in_train",
        "confirm_object_names_in_dev",
        "confirm_place_names_in_dev",
        "train_object_names_in_dev",
        "train_place_names_in_dev",
    }
    if set(proofs["zero_occurrence_keys"]) != expected_zero:
        raise RuntimeError("occurrence proof set drift")

    write_json(OUTPUT, payload)
    data_sha = sha256_file(OUTPUT)
    manifest = {
        "schema": "taiji-data-premise-18-0-manifest-v1",
        "step": "N18.0",
        "date": "2026-11-04",
        "generated_by": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(MODULE_PATH),
        "data_file": {
            "path": str(OUTPUT.relative_to(ROOT)),
            "sha256": data_sha,
            "bytes": OUTPUT.stat().st_size,
        },
        "digest": payload["digest"],
        "vocabulary": {
            "object_names": validation["object_names"],
            "place_names": validation["place_names"],
            "split_sizes": payload["split_sizes"],
            "name_counts": proofs["name_counts"],
            "object_name_bytes": validation["object_name_bytes"],
            "place_name_bytes": validation["place_name_bytes"],
        },
        "stats": stats,
        "proofs": proofs,
        "constraints": {
            "unseen_byte_strings": True,
            "combination_held_out": True,
            "pool_minimum": 200,
            "pool_actual": {"object_names": 220, "place_names": 220},
            "confirm_examples_generated": 0,
        },
        "capacity": capacity_note(stats),
        "cost": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "confirm_reads": 0,
            "confirm_examples_generated": 0,
        },
    }
    write_json(MANIFEST, manifest)
    write_json(PREREG, prereg(payload, stats, proofs, data_sha))
    print(
        json.dumps(
            {
                "data": str(OUTPUT.relative_to(ROOT)),
                "data_sha256": data_sha[:16],
                "data_bytes": manifest["data_file"]["bytes"],
                "train_examples": stats["train_examples"],
                "dev_examples": stats["dev_examples"],
                "confirm_examples_sealed": stats["confirm_examples"],
                "object_names": validation["object_names"],
                "place_names": validation["place_names"],
                "zero_occurrence_keys": len(proofs["zero_occurrence_keys"]),
                "combination_intersection": proofs["combination_intersection_train_dev"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
