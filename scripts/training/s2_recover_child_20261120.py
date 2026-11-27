"""Reload a saved S2 joint-gate child in an independent process and decode probes.

Used by the joint-gate recovery leg: the parent process saves the child state
dict, this process rebuilds the model from the recorded config, loads strictly,
and emits the same-condition decodes as hex strings for a bitwise comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5_v2 import generate_examples as generate_v2  # noqa: E402
from taiji.u2_22_5_pipeline import (  # noqa: E402
    DEFAULT_TEMPLATES,
    build_b_items,
    decode_episode,
    make_record,
)
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    model = U2Subject(U2Config(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model._verification_cuts = frozenset()
    fit = generate_v2("fit", 32)
    items = build_b_items(fit, templates=DEFAULT_TEMPLATES[:2])
    readings = []
    with torch.no_grad():
        for index in range(4):
            item = items[index]
            ex = item["example"]
            rec = make_record(item["records"][0], f"joint-recover-{index}", version=1)
            a_out, a_eos = decode_episode(model, ex["public"], ex["id"], records=None)
            b_out, b_eos = decode_episode(model, ex["public"], ex["id"], records=[rec])
            readings.append(
                {
                    "index": index,
                    "a": a_out.hex(),
                    "a_eos": a_eos,
                    "b": b_out.hex(),
                    "b_eos": b_eos,
                }
            )
    report = {"seed": args.seed, "verdict": "recover_ready", "readings": readings}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"seed": args.seed, "verdict": "recover_ready", "n": len(readings)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
