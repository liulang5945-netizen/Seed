"""Run the fixture fairness rules against the goal->action fixtures in this repo.

No model, no training, no data reads beyond importing the fixture modules. The point is
to make the 14.1/14.2 lesson executable: a fixture that writes the answer into the
prompt must fail a check before someone spends a run on it.

Fixtures under test are the live ones:

* ``taiji/goal_conditioned_training.py`` -- the 10.7f face, still imported by 10.7f and
  by the 14.2 rerun;
* ``taiji/native_goal_world_wide.py`` -- the 10.15 wide fixture;
* ``taiji/native_composition_fixture.py`` -- the 10.21 leak-free composition fixture.
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

from taiji.goal_conditioned_training import samples as legacy_samples  # noqa: E402
from taiji.goal_fixture_fairness import check_fixture, check_split  # noqa: E402
from taiji.native_composition_fixture import build_split  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    GLOSSARY,  # noqa: E402
    action_sentence,
    glossary_records,
)
from taiji.native_goal_world_wide import OBJECTS as WIDE_OBJECTS  # noqa: E402
from taiji.native_goal_world_wide import PLACES as WIDE_PLACES  # noqa: E402
from taiji.native_goal_world_wide import samples as wide_samples  # noqa: E402

BUDGET_CPU_SECONDS = 60
BUDGET_WALL_SECONDS = 180


def rotate(pairs):
    """A cyclic rotation is a bijection with no fixed point when the size is > 1."""
    values = [value for _, value in pairs]
    return [(key, values[(index + 1) % len(values)]) for index, (key, _) in enumerate(pairs)]


def legacy_fixture():
    """The 10.7f face: the goal *is* the answer sentence and there is no glossary."""
    records = ["桌面上有一个蓝色杯子和一个红色印章。", "蓝色杯子在窗台，红色印章在左盒。"]
    rows = [
        {
            "goal": sample["goal"],
            "action": action_for(sample),
            "answer_entities": (sample["correct"][0], sample["correct"][1]),
            "holdout_entities": (sample["correct"][1],),
            "records": records,
            # 10.7f took rows[:6] as train and rows[6:] as holdout, in fixture order.
            "split": "train" if index < 6 else "dev",
        }
        for index, sample in enumerate(legacy_samples())
    ]
    return rows, [*OBJECTS, *PLACES], None, None, []


def action_for(sample: dict) -> str:
    obj, place = sample["correct"]
    return f"把{obj}放到{place}"


def wide_fixture():
    lines = [text for text, _ in glossary_records()]
    rows = [
        {
            "goal": sample["goal"],
            "action": action_sentence(*sample["correct"]),
            "answer_entities": sample["correct"],
            "records": lines,
        }
        for sample in wide_samples()
    ]
    real = list(GLOSSARY.items())
    return rows, [*WIDE_OBJECTS, *WIDE_PLACES], rotate(real), real, []


def composition_fixture():
    """The 10.21 fixture declares its holdout inside each world by example layer, not by
    world id: ``seen_combination`` examples are the fitted face and
    ``unseen_combination``/``unseen_entity`` are the held-out ones. Splitting on world id
    would measure an axis the fixture never claimed, so the labels come from the layers.
    Only the entity layer promises a new entity, so only it contributes holdout entities."""
    rows, worlds, world_splits = [], [], []
    for split in ("train", "dev"):
        for built in build_split(split, 2):
            worlds.append(built)
            for example in built.examples:
                held_out_entity = example.layer == "unseen_entity"
                rows.append(
                    {
                        "goal": example.goal,
                        "action": example.action,
                        "answer_entities": (example.object_text, example.place_text),
                        "holdout_entities": (example.object_text,) if held_out_entity else (),
                        "records": [record.raw.decode() for record in example.records],
                        "records_signature": f"{example.world_index}|{example.layer}",
                        "split": "train" if example.layer == "seen_combination" else "dev",
                    }
                )
                world_splits.append(split)
    entities: list[str] = []
    for row in rows:
        for name in row["answer_entities"]:
            if name not in entities:
                entities.append(name)
    first = worlds[0]
    real = list(zip(first.object_descriptions, first.object_names, strict=True))
    # Same rows read on the axis the fixture does not claim: train worlds against dev
    # worlds. Dev worlds reuse three of the four object names, so that split is not an
    # entity holdout -- reported as an observation, never used to gate.
    world_rows = [
        dict(row, split=split, holdout_entities=row["answer_entities"][:1])
        for row, split in zip(rows, world_splits, strict=True)
    ]
    observation = check_split(world_rows)
    observation = {
        "rule": "world_axis_split_is_not_an_entity_holdout",
        "passed": observation["passed"],
        "observed": observation["observed"],
    }
    return rows, entities, rotate(real), real, [observation]


def summarise(name: str, rows: list[dict], checks: list[dict], observations=None) -> dict:
    failed = [check["rule"] for check in checks if check["passed"] is False]
    not_applied = [check["rule"] for check in checks if check["passed"] is None]
    return {
        "fixture": name,
        "samples": len(rows),
        "rules_checked": len(checks),
        "failed": failed,
        "not_applicable": not_applied,
        "fair": not failed,
        "checks": checks,
        "observations_not_gating": list(observations or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("fairness audit card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    torch_threads = None
    try:
        import torch

        torch.set_num_threads(1)
        torch_threads = torch.get_num_threads()
    except Exception:  # pragma: no cover - the audit must still run without torch
        pass

    fixtures = []
    for loader in (legacy_fixture, wide_fixture, composition_fixture):
        rows, entities, control, real, observations = loader()
        checks = check_fixture(
            rows, entities=entities, splits=rows, control=control, real_mapping=real
        )
        fixtures.append(summarise(loader.__name__, rows, checks, observations))

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    unfair = [item["fixture"] for item in fixtures if not item["fair"]]
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif unfair:
        status = "fixture_fairness_violation"
    else:
        status = "all_fixtures_fair"

    report = {
        "schema": "taiji-fixture-fairness-audit-v1",
        "step": "audit",
        "date": "2026-11-02",
        "status": status,
        "rules": [
            "goal_not_identical_to_action",
            "goal_names_no_answer_entity",
            "goal_shares_no_bigram_with_action",
            "goal_carries_no_action_template",
            "mapping_only_reachable_through_records",
            "split_holds_out_entities_and_novel_goals",
            "control_is_fixed_point_free_permutation",
        ],
        "fixtures": fixtures,
        "unfair_fixtures": unfair,
        "model_involved": False,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "String-level checks only: a fixture passing every rule is not shown to be "
            "learnable, and a fixture failing one rule is not shown to be useless -- it only "
            "means a copy shortcut exists on that face, so no selection conclusion may be "
            "drawn from it. The legacy face is listed because goal_conditioned_training is "
            "still imported by the 10.7f and 14.2 runners. The 10.21 face samples the first "
            "two train and two dev worlds of the declared 16/8, not the whole split."
        ),
        "torch_threads": torch_threads,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/goal_fixture_fairness.py",
                "taiji/native_goal_world_wide.py",
                "taiji/native_goal_world.py",
                "taiji/goal_conditioned_training.py",
                "taiji/native_composition_fixture.py",
                "scripts/training/audit_taiji_goal_fixture_fairness.py",
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
                "fixtures": [
                    {
                        "name": item["fixture"],
                        "samples": item["samples"],
                        "failed": item["failed"],
                        "not_applicable": item["not_applicable"],
                    }
                    for item in fixtures
                ],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
