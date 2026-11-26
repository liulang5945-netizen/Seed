"""10.19: let the subject *generate* the action instead of scoring it.

10.18 showed the subject can copy an entity name out of records (+3.1 nats, 5/5), and
10.17 showed a state-feature readout does not compose. So ask the subject itself:

    prompt   = "目标是让<obj descriptor>到<place descriptor>。我应该"
    records  = glossary
    decode   = greedy, 16 bytes

A hit means the continuation names the right object and the right place, in that order.
Nothing is trained; the model's own probabilities choose every byte.
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

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.native_goal_attributes import OBJECTS, PLACES  # noqa: E402
from taiji.native_goal_glossary import glossary_records  # noqa: E402

BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
MAX_NEW = 16
PASS_HITS = 5
ABSENT_HITS = 2

OBJECT_DESCRIPTORS = {
    "蓝色杯子": "圆口的器皿",
    "红色印章": "带纹路的方块",
}
PLACE_DESCRIPTORS = {
    "木桌": "四方木台",
    "窗台": "靠窗的台面",
    "左盒": "手边的匣子",
}


def decode(subject, provider, prompt: str, goal: str, records, subject_id: str) -> str:
    state = subject.initial_state(subject_id)
    index = 0
    gates = provider
    with torch.no_grad():
        for patch in subject.encoder.patches(prompt.encode()):
            output = subject.step(
                state,
                event=patch,
                goal=goal.encode(),
                records=records,
                growth_gates=gates(index),
            )
            state = output.state
            index += 1
        previous = EOS
        produced = []
        for _ in range(MAX_NEW):
            event = (
                subject.W.new_zeros(subject.config.width)
                if previous == EOS
                else subject.encoder.patches(bytes([previous]))[0]
            )
            output = subject.step(
                state,
                event=event,
                goal=goal.encode(),
                records=records,
                previous_byte=previous,
                growth_gates=gates(index),
            )
            byte = int(torch.argmax(output.probabilities))
            if byte == EOS:
                break
            produced.append(byte)
            state = subject.consume_output(output, byte)
            previous = byte
            index += 1
    return bytes(produced).decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.19 card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    records = glossary_records()

    tasks = []
    for obj in OBJECTS:
        for place in PLACES:
            tasks.append(
                {
                    "obj": obj,
                    "place": place,
                    "goal": f"让{OBJECT_DESCRIPTORS[obj]}到{PLACE_DESCRIPTORS[place]}。",
                }
            )

    rows = []
    hits = 0
    hits_without = 0
    for index, task in enumerate(tasks):
        prompt = f"目标是{task['goal']}我应该"
        goal_text = task["goal"]
        text = decode(subject, provider, prompt, goal_text, records, f"g{index}")
        plain = decode(subject, provider, prompt, goal_text, (), f"p{index}")
        ordered = text.find(task["obj"]) != -1 and text.find(task["place"]) > text.find(task["obj"])
        ordered_plain = plain.find(task["obj"]) != -1 and plain.find(task["place"]) > plain.find(
            task["obj"]
        )
        hits += int(ordered)
        hits_without += int(ordered_plain)
        rows.append(
            {
                "goal": goal_text,
                "expected": f"把{task['obj']}放到{task['place']}",
                "generated": text,
                "generated_without_records": plain,
                "hit": ordered,
            }
        )

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif hits >= PASS_HITS:
        status = "generation_selects"
    elif hits <= ABSENT_HITS:
        status = "generation_absent"
    else:
        status = "generation_partial"

    report = {
        "schema": "taiji-generation-10-19-v1",
        "step": "10.19",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "goals": len(tasks),
            "decode": "greedy, 16 bytes",
            "hit_rule": "continuation names the goal's object before its place",
            "pass_hits": PASS_HITS,
            "absent_hits": ABSENT_HITS,
            "control": "same prompt with no records",
            "training": "none",
        },
        "parent_model_digest": digest,
        "hits": hits,
        "hits_without_records": hits_without,
        "total": len(tasks),
        "rows": rows,
        "model_involved": True,
        "capability_claimed": status == "generation_selects",
        "training_admitted": False,
        "limits": (
            "Free generation only: a hit means the subject's own greedy continuation names "
            "the right object before the right place. It is one synthetic fixture, not P2, "
            "not natural-language capability, and it traains nothing."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_glossary.py",
                "scripts/training/verify_taiji_generation_10_19.py",
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
                "hits": hits,
                "hits_without_records": hits_without,
                "total": len(tasks),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
