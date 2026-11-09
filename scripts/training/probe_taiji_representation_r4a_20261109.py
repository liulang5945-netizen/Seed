"""R4a: does the representation suffice once the description/name gap is bridged?

R3 showed the failure is not encoder strength but a missing map between an entity
description ("圆口的器皿") and an entity name ("蓝色杯子"): within-family paraphrases
are fully separated, cross-family pairs are not separated at all.  This probe asks the
next question only: **if the bridge were perfect, would the project's own
representation separate the pairs?**

The bridge here is an oracle.  It replaces each glossary description with its entity
name using the fixture's own ``GLOSSARY``; no retrieval, no learning, no new parameter.
That is exactly its purpose -- it prices the bridge.  It is not evidence of any
capability and cannot upgrade ``representation_not_supported``.

Same 80 sentences, same 120/1,920 pairing, same distributions and the same thresholds
as the frozen report.  Only the text fed to the representation changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_representation_20261109 as frozen  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import GLOSSARY  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_representation_r4a_20261109.json"
REPORT = ROOT / "reports/taiji_representation_r4a_20261109.json"
PROGRESS = ROOT / "reports/taiji_representation_r4a_20261109_progress.jsonl"
ARMS = ("R1", "R2", "B")

DEFINITION = {
    "bridge": "oracle: replace every glossary description with its entity name, longest match first",
    "oracle": True,
    "claim": "upper bound only: it prices the bridge, it does not demonstrate a capability",
    "fixture": "unchanged 80 sentences / 20 goals / 4 variants from the frozen runner",
    "pairs": "unchanged 120 same / 1,920 unrelated / 1,120 near (near printed, never judged)",
}
CRITERIA = {
    "per_arm": (
        "separates iff gap>0 and AUC>=0.95 and 1-NN>=0.90; "
        "overlaps iff gap<=0 or AUC<0.90; otherwise inconclusive"
    ),
    "overall": (
        "bridge_sufficient if R1 or R2 separates; "
        "bridge_insufficient if R1 and R2 both overlap; otherwise inconclusive"
    ),
    "thresholds": "identical to SPEC section 21.4; this probe introduces no new threshold",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_progress(entry: dict[str, Any]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def normalise(text: str) -> str:
    """Replace description strings by entity names, longest description first."""
    for description, entity in sorted(GLOSSARY.items(), key=lambda item: (-len(item[0]), item[0])):
        text = text.replace(description, entity)
    return text


def bridged_rows() -> list[dict[str, Any]]:
    rows = frozen.build_fixture()
    out: list[dict[str, Any]] = []
    for row in rows:
        raw = row["text"]
        out.append({**row, "raw": raw, "text": normalise(raw)})
    return out


def check_fixture(rows: list[dict[str, Any]]) -> dict[str, Any]:
    goals = {row["goal"] for row in rows}
    per_goal = {goal: 0 for goal in goals}
    changed = {"describe": 0, "paraphrase": 0}
    unchanged_describe: list[str] = []
    changed_paraphrase: list[str] = []
    for row in rows:
        per_goal[row["goal"]] += 1
        if row["text"] != row["raw"]:
            changed["describe"] += 1
            if row["variant"] != "describe":
                changed_paraphrase.append(row["raw"])
        elif row["variant"] == "describe":
            unchanged_describe.append(row["raw"])
    return {
        "sentences": len(rows),
        "goals": len(goals),
        "variants_per_goal": sorted(set(per_goal.values())),
        "changed_by_bridge": changed,
        "describe_left_unchanged": unchanged_describe,
        "paraphrase_changed": changed_paraphrase,
    }


def phase_freeze() -> dict[str, Any]:
    rows = bridged_rows()
    same, unrelated, near = frozen.build_pairs(rows)
    check = check_fixture(rows)
    manifest = {
        "schema": "taiji-representation-r4a-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": "oracle_bridge_upper_bound_only",
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "fixture_sha256": frozen.fixture_hashes(),
        "counts": {
            "sentences": len(rows),
            "goals": check["goals"],
            "same_pairs": len(same),
            "unrelated_pairs": len(unrelated),
            "near_pairs_excluded_from_verdict": len(near),
        },
        "normalisation": {
            "rule": DEFINITION["bridge"],
            "glossary_entries": len(GLOSSARY),
            "description_bytes": sorted({len(text.encode("utf-8")) for text in GLOSSARY}),
            "records_below_min_addressable_16": sum(
                1 for text in GLOSSARY if len(text.encode("utf-8")) < 16
            ),
            "examples": [
                {"raw": row["raw"], "bridged": row["text"]}
                for row in rows
                if row["variant"] == "describe"
            ][:3],
        },
        "checks": check,
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "cost_estimate": "same shape as the frozen report: 80 sentences x 2 small encoders",
    }
    write_json(MANIFEST, manifest)
    return manifest


def arm_reading(matrix: list[list[float]], rows: list[dict[str, Any]], same, unrelated) -> dict:
    same_values = [matrix[left][right] for left, right in same]
    unrelated_values = [matrix[left][right] for left, right in unrelated]
    reading = frozen.distance_reading(same_values, unrelated_values)
    reading["nearest_neighbour"] = frozen.nearest_neighbour(matrix, [row["goal"] for row in rows])
    reading["verdict"] = frozen.classify(reading, reading["nearest_neighbour"]["accuracy"])
    reading["pairs"] = {
        "same": [
            {"i": left, "j": right, "distance": round(matrix[left][right], 6)}
            for left, right in same
        ],
        "unrelated": [
            {"i": left, "j": right, "distance": round(matrix[left][right], 6)}
            for left, right in unrelated
        ],
    }
    return reading


def phase_run(manifest: dict[str, Any]) -> dict[str, Any]:
    started = time.process_time()
    started_wall = time.time()
    rows = bridged_rows()
    same, unrelated, near = frozen.build_pairs(rows)
    check = check_fixture(rows)
    if check != manifest["checks"]:
        raise SystemExit("fixture or bridge changed after the freeze")
    if sha256_file(Path(__file__).resolve()) != manifest["runner_sha256"]:
        raise SystemExit("runner changed after the freeze")

    texts = [row["text"] for row in rows]
    reference = LivingCoreReference()
    arms: dict[str, Any] = {}
    for name in ARMS:
        if name == "B":
            matrix = [[frozen.bigram_distance(left, right) for right in texts] for left in texts]
        else:
            vectors = frozen.representation(name, texts, reference)
            matrix = frozen.cosine_matrix(vectors)
        arms[name] = arm_reading(matrix, rows, same, unrelated)
        append_progress({"phase": "arm_done", "arm": name, "verdict": arms[name]["verdict"]})

    separating = [name for name in ("R1", "R2") if arms[name]["verdict"] == "separates"]
    overlapping = [name for name in ("R1", "R2") if arms[name]["verdict"] == "overlaps"]
    if separating:
        overall = "bridge_sufficient"
    elif len(overlapping) == 2:
        overall = "bridge_insufficient"
    else:
        overall = "inconclusive"

    payload = {
        "schema": "taiji-representation-r4a-v1",
        "date": "2026-11-09",
        "arm": "R4a",
        "role": "oracle_bridge_upper_bound_only",
        "verdict": overall,
        "separating_arms": separating,
        "overlapping_arms": overlapping,
        "frozen_report": "reports/taiji_representation_20261109.json",
        "frozen_report_sha256": sha256_file(ROOT / "reports/taiji_representation_20261109.json"),
        "r3_report_sha256": sha256_file(ROOT / "reports/taiji_representation_20261109_r3.json"),
        "runner_sha256": manifest["runner_sha256"],
        "counts": manifest["counts"],
        "normalisation": manifest["normalisation"],
        "checks": check,
        "arms": arms,
        "disclosures": [
            "The bridge is an oracle: it uses the fixture's own GLOSSARY and takes no "
            "retrieval step. A positive result prices the bridge, it does not demonstrate one.",
            "It cannot upgrade representation_not_supported; the frozen decision arms are unchanged.",
            "The 9 glossary descriptions are 12-18 bytes; 7 of them are shorter than the frozen "
            "16-byte minimum addressable entry, so the bridge cannot be built from byte anchors "
            "without changing the entry shape.",
        ],
        "cost": {
            "cpu_seconds": round(time.process_time() - started, 3),
            "wall_seconds": round(time.time() - started_wall, 2),
            "cpu_cap": frozen.CPU_CAP,
        },
    }
    write_json(REPORT, payload)
    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="write the manifest and exit")
    args = parser.parse_args(argv)
    frozen.acquire_lock()
    try:
        if args.freeze:
            manifest = phase_freeze()
            print(json.dumps(manifest["counts"], ensure_ascii=False))
            print(json.dumps(manifest["checks"], ensure_ascii=False))
            return 0
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload = phase_run(manifest)
        print(
            json.dumps(
                {
                    "verdict": payload["verdict"],
                    "separating": payload["separating_arms"],
                    "overlapping": payload["overlapping_arms"],
                    "R1": {
                        "gap": round(payload["arms"]["R1"]["gap_p10_unrelated_minus_p90_same"], 6),
                        "auc": round(payload["arms"]["R1"]["auc"], 6),
                        "nn": payload["arms"]["R1"]["nearest_neighbour"]["accuracy"],
                    },
                    "R2": {
                        "gap": round(payload["arms"]["R2"]["gap_p10_unrelated_minus_p90_same"], 6),
                        "auc": round(payload["arms"]["R2"]["auc"], 6),
                        "nn": payload["arms"]["R2"]["nearest_neighbour"]["accuracy"],
                    },
                    "cost": payload["cost"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        frozen.release_lock()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
