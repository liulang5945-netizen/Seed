"""FEC-1 pre-registration guard.

Proves, in one run, that three things are real and not decorative:

1. ``criteria_sha256`` in the manifest equals the SHA-256 of canonical JSON over
   ``criteria`` (green), and a one-gate mutation is detected (red).
2. The three headline sentences (MAINLINE / MASTER-PLAN / ROADMAP-NEXT) are
   byte-identical (green), and a one-character mutation is detected (red).
3. The SPEC does not restate a second set of gate values, so it cannot drift
   away from the manifest (green), and a planted numeric drift is detected (red).

No graph is loaded, no checkpoint is read, no tensor is created.  This is a
document-and-hash guard only, which is why it may run before execution approval.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = PROJECT_ROOT / "plans/manifests/taiji_f_expressive_capacity_prereg_20261108.json"
SPEC = PROJECT_ROOT / "plans/reference/SPEC-F_EXPRESSIVE_CAPACITY_PREREG_20261108.md"
HEADLINE_FILES = (
    PROJECT_ROOT / "plans/MAINLINE.md",
    PROJECT_ROOT / "plans/MASTER-PLAN.md",
    PROJECT_ROOT / "plans/reference/ROADMAP-NEXT_20261104.md",
)
HEADLINE_MARKER = "**队首（唯一）**："


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def headline_of(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        index = line.find(HEADLINE_MARKER)
        if index >= 0:
            return line[index + len(HEADLINE_MARKER) :].strip()
    raise AssertionError(f"headline marker missing in {path}")


def check_criteria() -> dict:
    manifest = read_json(MANIFEST)
    declared = manifest["criteria_sha256"]
    actual = canonical_sha(manifest["criteria"])
    mutated = copy.deepcopy(manifest["criteria"])
    mutated["gates"]["key_effective_rank_min"] = 4
    mutated_sha = canonical_sha(mutated)
    return {
        "declared": declared,
        "recomputed": actual,
        "green_agrees": declared == actual,
        "red_mutation_changes_sha": mutated_sha != declared,
        "mutated_sha": mutated_sha,
    }


def all_identical(texts: list[str]) -> bool:
    """The single comparison used by both the green and the red exercise.

    Keeping one function for both directions is the point: if this function were
    a constant ``True``, the red exercise below would fail, so the guard cannot
    be decorative.
    """
    return len(set(texts)) == 1


def check_headline() -> dict:
    sentences = {path.name: headline_of(path) for path in HEADLINE_FILES}
    values = list(sentences.values())
    green = all_identical(values)
    probe = list(values)
    tail = probe[1][-1]
    probe[1] = probe[1][:-1] + ("，" if tail == "。" else "。")
    red = all_identical(probe) is False
    return {
        "green_all_identical": green,
        "red_mutation_detected": red,
        "comparator": "all_identical(texts) == len(set(texts)) == 1, shared by both directions",
        "lengths": {name: len(text) for name, text in sentences.items()},
        "sha256": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in sentences.items()
        },
        "headline": values[0],
    }


def check_spec_does_not_restate_gates() -> dict:
    """The SPEC must not carry its own copy of the gate numbers.

    It may name gates in prose, but any line that pairs a gate key with a numeric
    literal would create a second, driftable source.  We assert there is none,
    and prove the detector fires on a planted line.
    """
    text = SPEC.read_text(encoding="utf-8")
    gate_keys = list(read_json(MANIFEST)["criteria"]["gates"].keys())
    pattern = re.compile("|".join(re.escape(key) for key in gate_keys))

    def offenders(blob: str) -> list[str]:
        hits = []
        for line in blob.splitlines():
            if not pattern.search(line):
                continue
            if re.search(r"(>=|<=|≥|≤)\s*[0-9]", line):
                hits.append(line.strip()[:120])
        return hits

    green = offenders(text) == []
    planted = text + "\nkey_effective_rank_min ≥ 3\n"
    red = offenders(planted) != []
    return {
        "green_spec_has_no_gate_literal": green,
        "red_planted_literal_detected": red,
        "offenders": offenders(text),
    }


def main() -> int:
    criteria = check_criteria()
    headline = check_headline()
    spec = check_spec_does_not_restate_gates()
    guards = {
        "criteria_sha_green": criteria["green_agrees"],
        "criteria_sha_red": criteria["red_mutation_changes_sha"],
        "headline_green": headline["green_all_identical"],
        "headline_red": headline["red_mutation_detected"],
        "spec_no_gate_literal_green": spec["green_spec_has_no_gate_literal"],
        "spec_no_gate_literal_red": spec["red_planted_literal_detected"],
    }
    passed = all(guards.values())
    receipt = {
        "schema": "taiji-fec-1-prereg-guard-v1",
        "date": "2026-11-08",
        "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "guards": guards,
        "passed": passed,
        "detail": {"criteria": criteria, "headline": headline, "spec": spec},
        "note": "document-and-hash guard only; 0 graph loads, 0 checkpoint reads, 0 forward passes",
    }
    out = PROJECT_ROOT / "reports/taiji_f_expressive_capacity_prereg_guard_20261108.json"
    out.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"passed": passed, "guards": guards}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
