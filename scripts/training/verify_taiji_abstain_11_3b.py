"""11.3b: does a confidence gate on the model's own distribution discriminate?

The entry got an abstain channel that reads only the raw distribution (top probability
< 0.5 or entropy > 3.0 nats); no text template is involved. This run asks whether that
gate separates the five answerable items from the two that need clarify/abstain.

A gate that never fires or always fires is reported as degenerate -- it would be a
channel that exists but carries no information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest  # noqa: E402
from taiji.candidate_session import CandidateSession  # noqa: E402
from taiji.native_task_suite_11_3 import TASKS, normalize  # noqa: E402

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
CONFIDENCE = 0.5
ENTROPY = 3.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.3b card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    bundle = ROOT / BUNDLE
    manifest = build_manifest(ROOT, bundle)

    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        session = CandidateSession(ROOT, manifest_path, bundle, "frozen_eval")
        rows = []
        abstain_on_scorable = 0
        abstain_on_unscorable = 0
        answered_scorable = 0
        exact_hits = 0
        for task in TASKS:
            session.observe(task["prompt"])
            verdict = session.answer_or_abstain(CONFIDENCE, ENTROPY)
            kind = verdict["kind"]
            content = verdict["content"]
            top = verdict["raw"]["top_byte"]
            text = ""
            if kind == "answer" and top not in (256,):
                try:
                    text = bytes([top]).decode("utf-8")
                except UnicodeDecodeError:
                    text = ""
            hit = None
            if task["scored"]:
                if kind == "abstain":
                    abstain_on_scorable += 1
                else:
                    answered_scorable += 1
                    hit = normalize(text) == normalize(task["expected"])
                    exact_hits += int(hit)
            elif kind == "abstain":
                abstain_on_unscorable += 1
            rows.append(
                {
                    "id": task["id"],
                    "category": task["category"],
                    "scored": task["scored"],
                    "kind": kind,
                    "content_byte": content,
                    "text": text,
                    "top_probability": verdict["raw"]["top_probability"],
                    "entropy": verdict["raw"]["entropy"],
                    "hit": hit,
                }
            )

    scorable = sum(1 for task in TASKS if task["scored"])
    unscorable = len(TASKS) - scorable
    degenerate = abstain_on_scorable + abstain_on_unscorable in (0, len(TASKS))
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif degenerate:
        status = "abstain_gate_degenerate"
    else:
        status = "abstain_gate_discriminates"

    report = {
        "schema": "taiji-abstain-11-3b-v1",
        "step": "11.3b",
        "date": "2026-11-01",
        "status": status,
        "candidate_id": manifest["candidate_id"],
        "gate": {"confidence": CONFIDENCE, "entropy": ENTROPY},
        "tasks": len(TASKS),
        "scorable": scorable,
        "unscorable": unscorable,
        "abstain_on_scorable": abstain_on_scorable,
        "abstain_on_unscorable": abstain_on_unscorable,
        "answered_scorable": answered_scorable,
        "exact_hits": exact_hits,
        "rows": rows,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Channel check only: the gate reads the model's own top probability and entropy, "
            "not any text. A discriminating gate says the channel carries information; it "
            "does not make the two items scorable in a capability sense."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/candidate_session.py",
                "taiji/native_task_suite_11_3.py",
                "scripts/training/verify_taiji_abstain_11_3b.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "abstain_on_scorable": abstain_on_scorable,
                "abstain_on_unscorable": abstain_on_unscorable,
                "exact_hits": exact_hits,
                "answered": answered_scorable,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
