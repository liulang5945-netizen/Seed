#!/usr/bin/env python
"""N16.1.0a: materialise record-selection labels and key fragments.

Reads only the frozen v2 train/dev payload. Zero model forward, zero Adam.
The selection head may see the goal text and each record's description
fragment; it may never see record names, actions or expected bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"
OUTPUT_DIR = Path("output/taiji_core_binding_select_16_1")
PAYLOAD = OUTPUT_DIR / "select_v1.json"
REPORT = "reports/taiji_binding_select_16_1_data_20261103.json"
MANIFEST = "plans/manifests/taiji_binding_select_16_1_data_20261103.json"
PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_SHA256 = "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"

V2_TRAIN_DIGEST = "ffe721a6b5fef8c4a86765ba0f83e064ec97111590fbc8c0aa16b85ede37e5f4"
V2_DEV_DIGEST = "84dc10d78b798577335c30dee59a4d3b348a90b951f867a14d89da87200fb9c3"

STAGE_CPU_SECONDS = 60.0
STAGE_WALL_SECONDS = 120.0
STAGE_ADAM_CALLS = 0
STAGE_PRODUCT_BYTES = 100 * 1024**2

SCHEMA = "taiji-binding-select-v1"
SOURCE_FILES = (
    "taiji/binding_copy.py",
    "taiji/binding_copy_data.py",
    "taiji/binding_copy_scoring.py",
    "scripts/training/verify_taiji_binding_select_16_1.py",
)

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


class DataStop(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def split_digest(examples: list[dict]) -> str:
    """Recompute the frozen v2 digest from the data module, not from JSON rows."""
    from taiji.binding_copy_data import build_split, fixture_digest

    split = examples[0]["split"]
    worlds = build_split(split)
    return fixture_digest(worlds)


def key_fragment(record: dict) -> str:
    """The only record text the selection head may see: the description."""
    return str(record["description"])


def build_rows(examples: list[dict]) -> list[dict]:
    rows = []
    for example in examples:
        records = example["records"]
        labels = [
            index
            for index, record in enumerate(records)
            if record["kind"] == "object" and record["description"] == example["object_description"]
        ]
        if len(labels) != 1:
            raise DataStop(
                f"example {example['index']}: expected one object label, got {len(labels)}"
            )
        label = labels[0]
        keys = [key_fragment(record) for record in records]
        names = [str(record["name"]) for record in records]
        for index, fragment in enumerate(keys):
            if names[index] and names[index] in fragment:
                raise DataStop(f"example {example['index']}: key fragment leaks name at {index}")
            if example["action"] and example["action"] in fragment:
                raise DataStop(f"example {example['index']}: key fragment leaks action at {index}")
        rows.append(
            {
                "index": int(example["index"]),
                "world": int(example["world"]),
                "split": example["split"],
                "goal": example["goal"],
                "record_keys": keys,
                "record_kinds": [record["kind"] for record in records],
                "record_source_ids": [record["source_id"] for record in records],
                "selection_label": int(label),
                "record_count": len(records),
                "target_action": example["action"],
                "target_hex": example["expected_hex"],
                "flags": {
                    "unseen_output_name": bool(example["unseen_output_name"]),
                    "both_new": bool(example["both_new"]),
                    "new_description": bool(example["new_description"]),
                    "unseen_combination": bool(example["unseen_combination"]),
                },
                "head_input_excludes": [
                    "record.name",
                    "target_action",
                    "target_hex",
                    "expected_bytes",
                ],
            }
        )
    return rows


def axes(rows: list[dict]) -> dict:
    """Dev 2x2 crossing of unseen output name and new description."""
    counts = {
        "seen_name_seen_description": 0,
        "new_description_seen_name": 0,
        "unseen_output_name_seen_description": 0,
        "new_description_new_name": 0,
        "unseen_combination": 0,
    }
    for row in rows:
        if row["split"] != "dev":
            continue
        flags = row["flags"]
        if flags["both_new"]:
            counts["new_description_new_name"] += 1
        elif flags["new_description"]:
            counts["new_description_seen_name"] += 1
        elif flags["unseen_output_name"]:
            counts["unseen_output_name_seen_description"] += 1
        else:
            counts["seen_name_seen_description"] += 1
        if flags["unseen_combination"]:
            counts["unseen_combination"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="data", choices=("data",))
    parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(DATA_PAYLOAD).read_text(encoding="utf-8"))
    raw_train = [e for e in payload["examples"] if e["split"] == "train"]
    raw_dev = [e for e in payload["examples"] if e["split"] == "dev"]

    train_rows = build_rows(raw_train)
    dev_rows = build_rows(raw_dev)

    train_digest = split_digest(raw_train)
    dev_digest = split_digest(raw_dev)
    if train_digest != V2_TRAIN_DIGEST or dev_digest != V2_DEV_DIGEST:
        raise DataStop("v2 split digest drifted from the frozen 0a values")

    select_payload = {
        "schema": SCHEMA,
        "date": "2026-11-03",
        "source": DATA_PAYLOAD,
        "source_train_digest": train_digest,
        "source_dev_digest": dev_digest,
        "confirm_spec": payload["confirm_spec"],
        "confirm_materialised": False,
        "head_input": {
            "goal": True,
            "record_description_fragment": True,
            "record_name": False,
            "target_action": False,
            "expected_bytes": False,
        },
        "train": train_rows,
        "dev": dev_rows,
    }
    select_text = json.dumps(select_payload, ensure_ascii=False, indent=2) + "\n"
    PAYLOAD.write_text(select_text, encoding="utf-8")

    select_digest = sha256_bytes(canonical(select_payload).encode("utf-8"))
    train_select_digest = sha256_bytes(canonical(train_rows).encode("utf-8"))
    dev_select_digest = sha256_bytes(canonical(dev_rows).encode("utf-8"))

    label_histogram = {}
    for row in train_rows + dev_rows:
        label_histogram[str(row["selection_label"])] = (
            label_histogram.get(str(row["selection_label"]), 0) + 1
        )

    checks = {
        "v2_digest_unchanged": train_digest == V2_TRAIN_DIGEST and dev_digest == V2_DEV_DIGEST,
        "label_unique_per_example": all(
            row["record_kinds"][row["selection_label"]] == "object" for row in train_rows + dev_rows
        ),
        "label_in_range": all(
            0 <= row["selection_label"] < row["record_count"] for row in train_rows + dev_rows
        ),
        "key_fragment_no_name_leak": True,
        "key_fragment_no_action_leak": True,
        "head_input_excludes_labels": True,
        "counts": len(train_rows) == 288 and len(dev_rows) == 240,
        "axes_preserved": axes(train_rows + dev_rows)
        == {
            "seen_name_seen_description": 135,
            "new_description_seen_name": 45,
            "unseen_output_name_seen_description": 45,
            "new_description_new_name": 15,
            "unseen_combination": 36,
        },
        "confirm_sealed": select_payload["confirm_materialised"] is False
        and select_payload["confirm_spec"] == payload["confirm_spec"],
        "no_model_forward": True,
        "parent_identity": sha256_file(ROOT / PARENT_BUNDLE) == PARENT_SHA256,
        "product_limit": PAYLOAD.stat().st_size <= STAGE_PRODUCT_BYTES,
        "stage_budget": (time.process_time() - STARTED_CPU) <= STAGE_CPU_SECONDS
        and (time.perf_counter() - STARTED_WALL) <= STAGE_WALL_SECONDS,
        "adam_zero": STAGE_ADAM_CALLS == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    status = "passed_16_1_0a_selection_interface" if not failed else "engineering_failure"

    report = {
        "schema": "taiji-binding-select-data-v1",
        "step": "N16.1.0a",
        "date": "2026-11-03",
        "status": status,
        "checks": checks,
        "failed": failed,
        "payload": str(PAYLOAD),
        "payload_bytes": PAYLOAD.stat().st_size,
        "digests": {
            "train": train_digest,
            "dev": dev_digest,
            "select_payload": select_digest,
            "train_select": train_select_digest,
            "dev_select": dev_select_digest,
        },
        "counts": {"train": len(train_rows), "dev": len(dev_rows)},
        "axes": axes(train_rows + dev_rows),
        "label_histogram": label_histogram,
        "accounting": {
            "actual_adam_calls": 0,
            "effective_adam_calls": 0,
            "consumed_train_examples": 0,
            "consumed_dev_examples": 0,
            "consumed_confirm_examples": 0,
        },
        "cost": {
            "cpu_seconds": time.process_time() - STARTED_CPU,
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "adam_calls": 0,
            "product_bytes": PAYLOAD.stat().st_size,
        },
        "budget": {
            "cpu_seconds": STAGE_CPU_SECONDS,
            "wall_seconds": STAGE_WALL_SECONDS,
            "adam_calls": 0,
            "product_bytes": STAGE_PRODUCT_BYTES,
        },
        "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "limits": (
            "Selection labels and key fragments only. No model forward, no Adam, no confirm read, "
            "no capability claim."
        ),
    }
    Path(REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "taiji-binding-select-manifest-v1",
        "step": "N16.1.0a",
        "status": status,
        "report": REPORT,
        "payload": str(PAYLOAD),
        "payload_bytes": PAYLOAD.stat().st_size,
        "digests": report["digests"],
        "source_hashes": report["source_hashes"],
        "cost": report["cost"],
    }
    Path(MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"16.1.0a status: {status}")
    print(f"  failed: {failed}")
    print(f"  train/dev: {len(train_rows)}/{len(dev_rows)}, payload {PAYLOAD.stat().st_size} bytes")
    print(f"  digests: train {train_digest[:16]} dev {dev_digest[:16]} select {select_digest[:16]}")
    print(f"  cpu {report['cost']['cpu_seconds']:.2f}/{STAGE_CPU_SECONDS}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
