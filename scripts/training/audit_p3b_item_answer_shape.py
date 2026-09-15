"""Does the P3a baseline's zero-scoring correlate with the length of the expected answer token?

If yes, an enlarged CAP-0 surface buys resolution only when it is stratified by answer length.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
R = Path("E:/Seed/reports")
PATHS = {
    "p3a_base": R / "taiji_cap0_baseline_constrained_20260915.json",
    "trt_s1": R / "p3b_stages/treatment/cap0_tick_17000000.json",
    "ctrl_s1": R / "p3b_stages/control/cap0_tick_17000000.json",
}


def chars(token: str) -> int:
    return len(token.strip())


for name, path in PATHS.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"===== {name} =====")
    for dim in ("C", "D", "E"):
        items = data["dimensions"][dim]["items"]
        buckets = {}
        for it in items:
            if not isinstance(it.get("score"), int):
                continue
            tokens = it.get("expected_contains") or []
            if not tokens:
                key = "no_token"
            else:
                shortest = min(chars(t) for t in tokens)
                key = "len<=2" if shortest <= 2 else ("len<=5" if shortest <= 5 else "len>5")
            row = buckets.setdefault(key, [0, 0, []])
            row[0] += 1
            row[1] += it["score"] == 1
            if it["score"] == 1:
                row[2].append(it["id"])
        for key in sorted(buckets):
            total, hits, ids = buckets[key]
            print(f"  {dim} {key:8s} items={total:2d} scored_1={hits:2d} {ids}")
