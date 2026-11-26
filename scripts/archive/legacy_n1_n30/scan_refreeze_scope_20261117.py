"""Scan every manifest for pins on ``taiji/u2_subject.py`` (read-only, no training).

A new readout switch changes that file, so the re-freeze scope must be known
before the wiring is written.  Pins live in four pools -- ``source_hashes``,
``source_guards``, ``exclusion_sources.*`` and ``construction.*`` -- and an
earlier one-liner missed all of them because it assumed the pinned path was the
dict *key* with the digest as value.  This script does not assume a shape: it
walks each manifest, records every leaf whose path or value mentions the file,
and classifies by the manifest's top-level key.  Each recorded digest is compared
against the file's current bytes so current-value and historical pins are
separated.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "taiji/u2_subject.py"
REPORT = ROOT / "reports/taiji_u2_22_5_a_refreeze_scan_20261117.json"
POOLS = ("source_hashes", "source_guards", "source_pins", "exclusion_sources", "construction")


def _walk(node, path: tuple, hits: list) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if "u2_subject" in str(key):
                hits.append({"path": [*path, "<key>"], "value": str(value), "via_key": str(key)})
            _walk(value, (*path, str(key)), hits)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, (*path, str(index)), hits)
    elif isinstance(node, str) and "u2_subject" in node:
        hits.append({"path": list(path), "value": node})


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    current = hashlib.sha256((ROOT / TARGET).read_bytes()).hexdigest()
    current_sig = hashlib.sha256(
        (ROOT / TARGET).read_text(encoding="utf-8-sig").encode()
    ).hexdigest()
    records = []
    for manifest in sorted((ROOT / "plans/manifests").glob("*.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        hits: list = []
        _walk(payload, (), hits)
        if not hits:
            continue
        pools = Counter()
        rows = []
        for hit in hits:
            top = hit["path"][0] if hit["path"] else "<root>"
            pool = (
                top
                if top in POOLS
                else ("exclusion_sources" if top.startswith("exclusion") else "other")
            )
            if not (len(hit["value"]) == 64 and all(c in "0123456789abcdef" for c in hit["value"])):
                pool = "text_mention"
            pools[pool] += 1
            value = hit["value"]
            rows.append(
                {
                    "pool": pool,
                    "path": "/".join(hit["path"][:6]),
                    "value": value if len(value) <= 80 else value[:80] + "...",
                    "is_digest": len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                    "matches_current_raw": value == current,
                    "matches_current_sig": value == current_sig,
                }
            )
        records.append(
            {
                "manifest": manifest.name,
                "pools": dict(pools),
                "hits": rows,
            }
        )
    summary = Counter()
    for record in records:
        for pool in record["pools"]:
            summary[pool] += 1
    payload = {
        "schema": "taiji-u2-22-5-a-refreeze-scan-v1",
        "card": "22.5.0.A",
        "target": TARGET,
        "current_sha256_raw": current,
        "current_sha256_utf8sig": current_sig,
        "manifests_scanned": len(list((ROOT / "plans/manifests").glob("*.json"))),
        "manifests_referencing_target": len(records),
        "manifests_by_pool": dict(summary),
        "records": records,
        "discipline": {
            "files_modified": 0,
            "u2_optimizer_steps": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
    }
    REPORT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "manifests="
        + str(len(records))
        + " by_pool="
        + json.dumps(dict(summary))
        + " digest_hits="
        + str(sum(1 for r in records for h in r["hits"] if h["is_digest"]))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
