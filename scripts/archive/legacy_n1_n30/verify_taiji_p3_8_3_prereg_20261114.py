"""Guard for the frozen P3.8.3 pre-registration. No graph, no data, no training."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_8_3_prereg_20261114.json"
SPEC = ROOT / "plans/reference/SPEC-TAIJI_U2_P3_8_3_RECIPE_COMPARISON_PREREG_20261114.md"
RECEIPT = ROOT / "reports/taiji_u2_p3_8_3_prereg_guard_20261114.json"


def canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def gate_literals(text: str) -> list[str]:
    return [
        line.strip()[:120]
        for line in text.splitlines()
        if re.search(r"(>=|≤|≥)\s*[0-9]", line)
        and re.search(r"gain|threshold|delta|minimum|cap|预算|门", line, re.IGNORECASE)
    ]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = manifest["criteria_sha256"]
    recomputed = canonical_sha(manifest["criteria"])
    mutated = copy.deepcopy(manifest["criteria"])
    mutated["gates"]["minimum_meaningful_gain"] = 0.5
    mutated_sha = canonical_sha(mutated)
    spec_text = SPEC.read_text(encoding="utf-8")
    offenders = gate_literals(spec_text)
    planted = gate_literals(spec_text + "\nminimum_meaningful_gain ≥ 0.9\n")
    guards = {
        "criteria_sha_green": declared == recomputed,
        "criteria_sha_red": mutated_sha != declared,
        "spec_no_gate_literal_green": offenders == [],
        "spec_no_gate_literal_red": planted != [],
        "price_report_cited": bool(manifest["price"]["report"]),
        "execution_runner_hash_null": manifest["execution_runner_sha256"] is None,
        "report_hash_null": manifest["report_sha256"] is None,
    }
    passed = all(guards.values())
    receipt = {
        "schema": "taiji-u2-p3-8-3-prereg-guard-v1",
        "date": "2026-11-14",
        "criteria_sha256": declared,
        "recomputed": recomputed,
        "mutated": mutated_sha,
        "guards": guards,
        "offenders": offenders,
        "passed": passed,
        "note": "document-and-hash guard only; 0 graph loads, 0 optimizer steps, 0 data reads",
    }
    RECEIPT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"passed": passed, "guards": guards}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
