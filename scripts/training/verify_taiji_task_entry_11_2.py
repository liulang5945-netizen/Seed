"""11.2: one task entry, raw output first, nothing filled in by the display layer.

Checks on the 09.8 candidate:

* ``answer`` keeps ``raw`` and ``decoded`` separate and ``raw`` comes first;
  ``decoded`` is at most one character taken from the raw top byte;
* ``clarify`` returns ``content=None`` -- it never fabricates an answer;
* ``reset`` restores the initial state digest exactly;
* two sessions replaying the same calls give bit-identical raw outputs;
* in ``frozen_eval`` a mutating call is refused.

Nothing is trained and the bundle is opened read-only.
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

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
PROMPT = "目标是让圆口的器皿到四方木台。"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.2 card is write-once")

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
        initial = session.initial_digest
        session.observe(PROMPT)
        after_observe = session.transcript[-1]["state"]
        answer = session.answer()
        clarify = session.clarify("目标未指明对象")
        reset = session.reset()

        replay = CandidateSession(ROOT, manifest_path, bundle, "frozen_eval")
        replay.observe(PROMPT)
        replay_answer = replay.answer()

        refused = session.refuse_mutation("learn")

    raw_keys = list(answer.keys())
    raw_first = raw_keys.index("raw") < raw_keys.index("decoded")
    decoded = answer["decoded"]
    decoded_ok = decoded is None or (
        isinstance(decoded, str) and len(decoded) == 1 and ord(decoded) == answer["raw"]["top_byte"]
    )
    observe_changed_state = after_observe != initial
    reset_ok = reset["state"] == initial
    deterministic = answer["raw"] == replay_answer["raw"]
    clarify_ok = clarify["content"] is None and clarify["kind"] == "clarify"
    refused_ok = refused["refused"] == "frozen_eval"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif not (raw_first and decoded_ok):
        status = "display_fabrication"
    elif not clarify_ok:
        status = "clarify_fabricates"
    elif not reset_ok or not observe_changed_state:
        status = "reset_mismatch"
    elif not deterministic:
        status = "nondeterministic_entry"
    elif not refused_ok:
        status = "frozen_mutation_allowed"
    else:
        status = "entry_contract_verified"

    report = {
        "schema": "taiji-task-entry-11-2-v1",
        "step": "11.2",
        "date": "2026-11-01",
        "status": status,
        "prompt": PROMPT,
        "candidate_id": manifest["candidate_id"],
        "mode": "frozen_eval",
        "raw_first": raw_first,
        "decoded_channel_ok": decoded_ok,
        "clarify_content": clarify["content"],
        "observe_changed_state": observe_changed_state,
        "reset_restores_initial": reset_ok,
        "two_sessions_identical": deterministic,
        "frozen_mutation_refused": refused_ok,
        "answer_raw": answer["raw"],
        "answer_decoded": decoded,
        "not_delivered": ["save", "load", "outcome", "act"],
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Entry contract only: it checks that raw output is stored first, that the "
            "decoded channel cannot invent content, and that the frozen mode refuses "
            "mutation. save/load/outcome/act are NOT delivered. It says nothing about "
            "answer quality."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/candidate_session.py",
                "scripts/training/verify_taiji_task_entry_11_2.py",
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
                "raw_first": raw_first,
                "decoded_ok": decoded_ok,
                "reset_ok": reset_ok,
                "deterministic": deterministic,
                "frozen_refused": refused_ok,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
