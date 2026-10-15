"""PLAN-N1-14 synthetic data and train-only no-update cost preflight."""

from __future__ import annotations

import hashlib
import json
import random
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training.audit_taiji_n1_position_evidence_reference import digest  # noqa: E402
from taiji.contracts import Observation  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402

KINDS = ("lookup", "revision", "transition", "unknown")
MATERIALS = {
    "train": ("copper", "silver", "iron", "brass"),
    "dev": ("nickel", "lead"),
    "holdout": ("quartz", "amber"),
}
DIRECTIONS = ("north", "east", "south", "west")


def build():
    rng = random.Random(14)
    episodes = []
    used_names = set()

    def entity():
        while True:
            name = "".join(rng.choices(string.ascii_lowercase, k=10))
            if name not in used_names:
                used_names.add(name)
                return name

    for split, count in (("train", 96), ("dev", 32), ("holdout", 32)):
        for index in range(count):
            kind = KINDS[index % 4]
            name, other, third = entity(), entity(), entity()
            first, second = rng.sample(MATERIALS[split], 2)
            if kind == "lookup":
                observations = [f"{name} uses {first}.", f"{other} uses {second}."]
                rng.shuffle(observations)
                question, answer = f"material of {name}?", first
            elif kind == "revision":
                observations = [f"{name} uses {first}.", f"{name} now uses {second}."]
                question, answer = f"material of {name}?", second
            elif kind == "transition":
                start = rng.randrange(4)
                turn = rng.choice((-1, 1))
                observations = [
                    f"{name} faces {DIRECTIONS[start]}.",
                    f"{name} turns {'right' if turn == 1 else 'left'}.",
                ]
                question, answer = f"direction of {name}?", DIRECTIONS[(start + turn) % 4]
            else:
                observations = [f"{other} uses {first}.", f"{third} uses {second}."]
                question, answer = f"material of {name}?", "unknown"
            episodes.append(
                {
                    "id": f"{split}-{index:03d}",
                    "split": split,
                    "kind": kind,
                    "entities": [name, other, third],
                    "observations": observations,
                    "question": question,
                    "answer": answer,
                }
            )
    return {"schema": "n1-native-evidence-curriculum-v2", "seed": 14, "episodes": episodes}


def validate(manifest):
    episodes = manifest["episodes"]
    assert len({item["id"] for item in episodes}) == len(episodes) == 160
    names = {}
    counts = {}
    for split, size in (("train", 96), ("dev", 32), ("holdout", 32)):
        subset = [item for item in episodes if item["split"] == split]
        assert len(subset) == size
        counts[split] = {kind: sum(item["kind"] == kind for item in subset) for kind in KINDS}
        assert set(counts[split].values()) == {size // 4}
        names[split] = {name for item in subset for name in item["entities"]}
    for left, right in (("train", "dev"), ("train", "holdout"), ("dev", "holdout")):
        assert names[left].isdisjoint(names[right])
    for item in episodes:
        assert len(item["observations"]) == 2
        assert all(
            len(name) == 10 and name.isalpha() and name.islower() for name in item["entities"]
        )
        assert item["answer"] not in item["question"]
        if item["kind"] == "transition":
            assert item["answer"] not in " ".join(item["observations"])
    return counts


def preflight(manifest):
    torch.set_num_threads(1)
    torch.manual_seed(14)
    model = NativeEvidenceCandidate()
    before = digest(model.state_dict().values())
    rows = []
    for kind in KINDS:
        item = next(e for e in manifest["episodes"] if e["split"] == "train" and e["kind"] == kind)
        for arm in ("position", "terminal", "cut"):
            torch.manual_seed(14)
            model = NativeEvidenceCandidate()
            assert before == digest(model.state_dict().values())
            for index, text in enumerate(item["observations"], 1):
                model.observe("preflight", Observation("text", text, index, f"source-{index}"))
            started = time.process_time()
            tape = model.evidence(
                "preflight", cut_credit=arm == "cut", terminal_per_source=arm == "terminal"
            )
            target = item["answer"].encode("utf-8")
            logits, _ = model.teacher_logits(
                tape, "preflight", item["question"].encode("utf-8"), target
            )
            torch.nn.functional.cross_entropy(logits, torch.tensor([*target, 256])).backward()
            assert before == digest(model.state_dict().values())
            rows.append(
                {
                    "kind": kind,
                    "arm": arm,
                    "id": item["id"],
                    "cpu_seconds": time.process_time() - started,
                    "first_credit": (
                        None if tape.states[0].grad is None else float(tape.states[0].grad.norm())
                    ),
                    "last_credit": (
                        None if tape.states[-1].grad is None else float(tape.states[-1].grad.norm())
                    ),
                }
            )
    assert before == digest(model.state_dict().values())
    cpu = time.process_time()
    return {
        "rows": rows,
        "parameters_unchanged": True,
        "optimizer_steps": 0,
        "cpu_seconds": cpu,
        "cpu_limit_seconds": 30,
        "passed": cpu <= 30,
        "training_cost_estimate_is_upper_bound": False,
    }


def main():
    path = ROOT / "plans/manifests/n1_native_evidence_curriculum_v2_20261015.json"
    report = ROOT / "reports/taiji_n1_native_evidence_curriculum_preflight_v2_20261015.json"
    if path.exists() or report.exists():
        raise FileExistsError("curriculum and preflight refuse overwrite")
    manifest = build()
    counts = validate(manifest)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    result = {
        "schema": "n1-curriculum-preflight-v1",
        "counts": counts,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capability_claim": False,
        "holdout_model_queries": 0,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        result.update(preflight(manifest))
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
