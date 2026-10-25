"""Summarize frozen N1-167 records without model or corpus access."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def cjk_accuracy(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """All bytes must match within a complete target CJK character under TF."""
    raw = bytes(row["target"] for row in rows)
    hits = count = index = 0
    while index < len(raw):
        lead = raw[index]
        width = 1 if lead < 128 else 2 if lead < 224 else 3 if lead < 240 else 4
        chunk = raw[index : index + width]
        if len(chunk) < width:
            break
        character = chunk.decode("utf-8")
        if "\u4e00" <= character <= "\u9fff":
            count += 1
            hits += int(
                all(rows[j]["raw_top1"] == rows[j]["target"] for j in range(index, index + width))
            )
        index += width
    return hits, count


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "complete" or len(report["records"]) != 48:
        raise ValueError("analysis requires the complete preregistered 48-cell matrix")
    endpoints = {}
    classifications = []
    for endpoint, metadata in report["endpoint_summaries"].items():
        groups = {}
        for mode in ("configured", "protected", "oracle_plan"):
            rows = [
                row
                for row in report["records"]
                if row["endpoint_id"] == endpoint
                and row["mode"] == mode
                and row["status"] == "complete"
            ]
            if not rows:
                groups[mode] = {"status": "not_applicable"}
                continue
            matches = [cjk_accuracy(row["tf"]["rows"]) for row in rows]
            hits, total = sum(x[0] for x in matches), sum(x[1] for x in matches)
            groups[mode] = {
                "status": "complete",
                "episodes": len(rows),
                **{
                    name: statistics.mean(row["tf"][name] for row in rows)
                    for name in ("top1_raw", "top1_masked", "nll_raw_bits", "nll_masked_bits")
                },
                "free_target_prefix_match": statistics.mean(
                    row["target_prefix_match"] for row in rows
                ),
                "cjk_tf_characters_correct": hits,
                "cjk_tf_characters": total,
                "cjk_tf_character_accuracy": hits / max(1, total),
                "readonly_training_path_accuracy": statistics.mean(
                    row["training_path_readonly"]["accuracy"] for row in rows
                ),
            }
        native = [
            row
            for row in report["records"]
            if row["endpoint_id"] == endpoint and row["mode"] == "configured"
        ]
        groups["prompt_swap"] = {
            "mean_js_bits_min": min(row["prompt_swap"]["mean_js_bits"] for row in native),
            "mean_js_bits_max": max(row["prompt_swap"]["mean_js_bits"] for row in native),
            "top1_agreement_min": min(row["prompt_swap"]["top1_agreement"] for row in native),
            "top1_agreement_max": max(row["prompt_swap"]["top1_agreement"] for row in native),
            "mean_nll_delta": statistics.mean(
                row["prompt_swap"]["nll_raw_bits_delta"] for row in native
            ),
        }
        for row in native:
            matching = {
                item["mode"]: item
                for item in report["records"]
                if item["endpoint_id"] == endpoint
                and item["episode_id"] == row["episode_id"]
                and item["status"] == "complete"
            }

            def rescue(mode, candidates=matching, reference=row):
                other = candidates.get(mode)
                return bool(
                    other
                    and other["tf"]["top1_raw"] - reference["tf"]["top1_raw"] >= 0.10
                    and reference["tf"]["nll_raw_bits"] - other["tf"]["nll_raw_bits"] >= 0.50
                )

            classifications.append(
                {
                    "endpoint_id": endpoint,
                    "episode_id": row["episode_id"],
                    "high_tf_low_rollout": row["tf"]["top1_raw"] >= 0.8
                    and row["target_prefix_match"] < 0.2,
                    "oracle_rescue": rescue("oracle_plan"),
                    "protected_rescue": rescue("protected"),
                    "prompt_insensitive": row["prompt_swap"]["mean_js_bits"] < 1e-6
                    and row["prompt_swap"]["top1_agreement"] >= 0.99,
                }
            )
        endpoints[endpoint] = {
            "read_only_verified": metadata["read_only_verified"],
            "modes": groups,
        }
    return {
        "schema": "taiji-n1-plan167-attribution-summary-v1",
        "status": "complete",
        "aggregation": "macro mean over four fixed training episodes; CJK character counts pooled",
        "cjk_metric_scope": "posthoc descriptive complete-character TF metric; not a new preregistered gate",
        "endpoints": endpoints,
        "preregistered_classifications": classifications,
        "counts": {
            key: sum(row[key] for row in classifications)
            for key in (
                "high_tf_low_rollout",
                "oracle_rescue",
                "protected_rescue",
                "prompt_insensitive",
            )
        },
        "scope": "four known train episodes; not held-out capability or recipe ranking",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    payload["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    payload["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(payload["counts"]))


if __name__ == "__main__":
    main()
