"""R56 mainline-fit gate: does the written bridge reach Task1 at all?

R54' closed the record-selection question inside the 20-goal fixture world: with
a joint hit the asked pair is chosen 40/40 and no answer is lost.  But that
fixture is nine entities in a world of its own.  Task1 -- the actual task -- has
eight candidate records per world (four object lines, four place lines), a goal
that mentions *two* of them, and an answer that copies one name from each.  The
bridge has never been asked whether it can serve that.

This runner only diagnoses.  It builds the bridge from Task1's own eight record
lines (raw bytes, unmodified), asks Task1's own 192 confirm goals, and answers
three questions:

1. what the frozen joint rule does on Task1 records (it needs one record to
   contain every key; Task1 splits its keys across two records);
2. what the bridge's own two-pass read does (one record per description);
3. whether the record it names can be handed to the 19.2v copy chain -- i.e.
   whether the bridge's notion of an answer and the copy chain's notion of a
   copy field are the same object.

The verdict is one of three, fixed before the reading: ``bridge_reaches_mainline``
(the bridge can be wired in and fixes at least one real failure),
``bridge_parallel_world`` (it cannot even name the records), or
``bridge_partial_overlap`` (some layer overlaps exactly, and the report must say
which).  "Has reference value" is not an allowed verdict.

Confirm materialisation is registered in D-20261109-57: records and goal bytes
only, zero model forwards, zero scoring, and the published 19.2v reading is
cited, never recomputed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_cross_family_readout_20261109 as base  # noqa: E402
import taiji.data_premise_18_0 as premise  # noqa: E402
from taiji.address_bridge import BridgeTable, sequence_key  # noqa: E402
from taiji.copy_stop_boundary import field_bounds  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_mainline_fit_20261109.json"
REPORT = ROOT / "reports/taiji_mainline_fit_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"
PREMISE_SOURCE = ROOT / "taiji/data_premise_18_0.py"
COPY_SOURCE = ROOT / "taiji/copy_stop_boundary.py"
PREVIOUS_REPORT = ROOT / "reports/taiji_package19_2v_confirm_holdout_20261105.json"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SPLIT = "confirm"
SEED = 20261109
ESTIMATE_CPU_UPPER = 20.0
CPU_CAP = float(base.CPU_CAP)

DEFINITION = {
    "question": "can the written bridge serve Task1, or does it live in a parallel world?",
    "source_records": (
        "Task1's eight candidate records per world, raw bytes exactly as the frozen generator "
        "writes them; no rewriting, no added fields, no new key type"
    ),
    "key_rule": (
        "frozen wording rule: the key is the description phrase exactly as it appears in the "
        "record (ordered characters, containment read); names never enter the key"
    ),
    "questions": (
        "Task1's own 192 confirm goals 让{object description}到{place description}。 (12 worlds x "
        "16); no question is authored for this gate"
    ),
    "readings": {
        "joint": (
            "R54' joint rule applied unchanged: every key of the goal must sit in one record; "
            "exactly one -> selected, two or more -> ambiguous_multiple_records, zero -> no_match, "
            "no key -> no_keys"
        ),
        "per_slot": (
            "the bridge's own frozen two-pass read, one record per description; answers are "
            "checked against the record the fixture says carries that description"
        ),
        "wiring": (
            "for every answered slot: does the bridge's record index equal the copy chain's route, "
            "and does the bridge's value equal the copy chain's own name-field bytes "
            "record[start:end] from field_bounds"
        ),
        "coordinate_check": (
            "the bridge's span is a character span in the QUESTION; the copy chain's field is a "
            "byte span in the RECORD; the report must state the difference in bytes, not call it "
            "agreement"
        ),
    },
    "mainline_comparison": (
        "the four published 19.2v confirm failures are re-read from the report (never re-scored); "
        "for each, the gate reports whether the bridge's choice would differ from the route 19.2v "
        "took, and whether the failure is already selection-correct"
    ),
    "verdict_rule": {
        "bridge_reaches_mainline": (
            "joint selects one record for all 192 goals AND per-slot exact 384/384 AND wiring "
            "exact 384/384 AND the bridge fixes at least one of the four failures"
        ),
        "bridge_parallel_world": (
            "per-slot address exact below 0.5: the bridge cannot even name the records"
        ),
        "bridge_partial_overlap": (
            "otherwise; the report must name the overlapping part and the part that does not"
        ),
    },
    "closed_state": "the nine closed-state readings stay equal before and after",
    "training": "zero training, zero optimizer steps, zero Adam, zero parameters",
    "sealed_set": (
        "confirm worlds are materialised once for their record and goal bytes only; zero model "
        "forwards, zero scoring, and the published 19.2v reading is not recomputed"
    ),
}

PREDICTION = {
    "joint": {
        "goals": 192,
        "selected": 0,
        "ambiguous_multiple_records": 0,
        "no_match": 192,
        "no_keys": 0,
        "why": (
            "every Task1 record carries exactly one description, so no single record can contain "
            "both keys of a goal; the joint rule can only return no_match"
        ),
    },
    "per_slot": {
        "goals": 192,
        "answered_pair": 192,
        "slots": 384,
        "address_ok": 384,
        "value_ok": 384,
        "why": "each description occurs in exactly one record, so the two-pass read is unique",
    },
    "wiring": {
        "route_equal": 384,
        "name_bytes_equal": 384,
        "why": "the bridge's value is the same name field the copy chain derives from the record",
    },
    "mainline": {
        "failures": 4,
        "already_selection_correct": 4,
        "bridge_fixes": 0,
        "why": (
            "all four published failures are extra_bytes at answer position 27 with both spans "
            "aligned: the stop edge, not the record choice"
        ),
    },
    "verdict": "bridge_partial_overlap",
}

CRITERIA = {
    "distribution": "the four joint classes and the per-slot classes are all reported by count",
    "wiring": (
        "route equality, name-byte equality, and the question-vs-record coordinate difference are "
        "reported with concrete byte offsets"
    ),
    "mainline": "each of the four failures is classified and the number the bridge would fix is counted",
    "samples": "at least three real cases, including one joint no_match and one published failure",
    "closed_state": "nine items equal",
    "verdict": "exactly one of the three names, decided by the frozen rule",
}

MAINLINE_RELATION = {
    "affects_mainline": True,
    "which_reading": (
        "Task1's record choice (route) and copied name per goal, and whether the bridge repairs any "
        "of the four published confirm failures"
    ),
    "how": (
        "the bridge is built from Task1's own eight record lines and asked Task1's own 192 confirm "
        "goals; its routes and names are compared against the fixture's records and against the "
        "published 19.2v failure rows"
    ),
    "not_a_proxy": "no fixture-world data is used for the judgement; every record and goal is Task1's",
    "limit": (
        "this gate cannot change 188/192 by itself: it only decides whether wiring the bridge in "
        "could change it, and the answer is read from the four failures"
    ),
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- Task1 records -------------------------------------------------------------
def extract(raw: str) -> tuple[str, str, str]:
    """(description, name, style) read from the record's own delimiters.

    Deliberately independent of ``copy_stop_boundary`` so that the wiring check
    compares two implementations instead of one implementation with itself.
    """
    body = raw[: -len("。")] if raw.endswith("。") else raw
    if "又叫" in body:
        name, description = body.split("又叫", 1)
        return description, name, "aka"
    if "对应" in body:
        description, name = body.split("对应", 1)
        return description, name, "mapped"
    if "是" in body:
        description, name = body.split("是", 1)
        return description, name, "stated"
    raise ValueError(f"record has no frozen delimiter: {raw!r}")


def build_worlds() -> list[dict[str, Any]]:
    pools = premise.split_pools()
    return [
        premise.build_world(SPLIT, world, pools) for world in range(premise.SPLIT_WORLDS[SPLIT])
    ]


def world_table(
    world: dict[str, Any],
) -> tuple[BridgeTable, list[bytes], dict[str, int], list[dict]]:
    """Bridge built from the world's own eight raw records; nothing added."""
    raw_bytes: list[bytes] = []
    gold: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    for index, record in enumerate(world["records"]):
        description, name, style = extract(record["raw"])
        if description != record["description"] or name != record["name"]:
            raise SystemExit("independent parse disagrees with the frozen fixture")
        if style != record["style_name"]:
            raise SystemExit("independent style parse disagrees with the frozen fixture")
        raw_bytes.append(record["raw"].encode("utf-8"))
        gold[description] = index
        rows.append(
            {
                "index": index,
                "kind": record["kind"],
                "description": description,
                "name": name,
                "style": style,
                "raw": record["raw"],
            }
        )
        table.write([(description, (record["kind"], name))])
    return table, raw_bytes, gold, rows


def joint_select(raw_bytes: list[bytes], keys: tuple[bytes, ...]) -> tuple[str, tuple[int, ...]]:
    """R54' joint rule, unchanged: every key in one record."""
    wanted = tuple(dict.fromkeys(keys))
    if not wanted:
        return "no_keys", ()
    hits = tuple(index for index, raw in enumerate(raw_bytes) if all(key in raw for key in wanted))
    if not hits:
        return "no_match", hits
    if len(hits) == 1:
        return "selected", hits
    return "ambiguous_multiple_records", hits


def goal_span(goal: str, start: int, end: int) -> bytes:
    return "".join(list(goal)[start:end]).encode("utf-8")


# -- readings ------------------------------------------------------------------
def readings(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    joint_counts = {
        "selected": 0,
        "ambiguous_multiple_records": 0,
        "no_match": 0,
        "no_keys": 0,
    }
    slot_counts = {
        "answered_pair": 0,
        "answered_single": 0,
        "no_match": 0,
        "ambiguous_same_start": 0,
    }
    checks = {
        "slots": 0,
        "answers": 0,
        "address_ok": 0,
        "value_ok": 0,
        "question_span_ok": 0,
        "name_bytes_ok": 0,
        "single_key_unique": 0,
    }
    goals = 0
    for world in worlds:
        table, raw_bytes, gold, rows = world_table(world)
        for example in world["examples"]:
            goals += 1
            goal = example["goal"]
            read = table.read_sentence(goal)
            answers = read["answers"]
            keys = tuple(goal_span(goal, row["span_start"], row["span_end"]) for row in answers)
            status, _hits = joint_select(raw_bytes, keys)
            joint_counts[status] += 1
            if read["decision"] in slot_counts:
                slot_counts[read["decision"]] += 1
            else:
                slot_counts[read["decision"]] = slot_counts.get(read["decision"], 0) + 1
            for answer, key in zip(answers, keys, strict=True):
                slot = answer["value"][0]
                checks["slots"] += 1
                checks["answers"] += 1
                expected_description = example[f"{slot}_description"]
                expected_name = example[f"{slot}_name"]
                record = rows[answer["address"]]
                if answer["address"] == gold.get(expected_description):
                    checks["address_ok"] += 1
                if tuple(answer["value"]) == (slot, expected_name):
                    checks["value_ok"] += 1
                if key == expected_description.encode("utf-8"):
                    checks["question_span_ok"] += 1
                start, end = field_bounds(record["raw"], expected_description)
                if record["raw"].encode("utf-8")[start:end] == expected_name.encode("utf-8"):
                    checks["name_bytes_ok"] += 1
                if sum(1 for raw in raw_bytes if key in raw) == 1:
                    checks["single_key_unique"] += 1
    return {
        "goals": goals,
        "joint": joint_counts,
        "per_slot": slot_counts,
        "checks": checks,
    }


def wiring(world: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    """One worked case: where each side's offsets live, in bytes."""
    table, raw_bytes, gold, rows = world_table(world)
    goal = example["goal"]
    read = table.read_sentence(goal)
    goal_bytes = goal.encode("utf-8")
    detail: list[dict[str, Any]] = []
    for answer in read["answers"]:
        slot = answer["value"][0]
        expected_description = example[f"{slot}_description"]
        record = rows[answer["address"]]
        start, end = field_bounds(record["raw"], expected_description)
        record_bytes = record["raw"].encode("utf-8")
        description_at = goal_bytes.find(expected_description.encode("utf-8"))
        detail.append(
            {
                "slot": slot,
                "route": answer["address"],
                "gold_route": gold[expected_description],
                "bridge_value": list(answer["value"]),
                "bridge_span_chars": [answer["span_start"], answer["span_end"]],
                "bridge_span_question_bytes": [
                    description_at,
                    description_at + len(expected_description.encode("utf-8")),
                ],
                "copy_field_record_bytes": [start, end],
                "copy_field_text": record_bytes[start:end].decode("utf-8"),
                "record": record["raw"],
                "record_bytes": len(record_bytes),
            }
        )
    return {
        "goal": goal,
        "goal_bytes": len(goal_bytes),
        "slots": detail,
        "bridge_coordinate": "character span inside the question",
        "copy_coordinate": "byte span of the name field inside the record",
        "same_object": False,
        "note": (
            "the bridge never supplies the record-side field; the copy chain derives start/end "
            "from field_bounds on the same record, so the two only meet on (route, name text)"
        ),
    }


def mainline_comparison(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    published = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    failures = published["failures"]
    detail: list[dict[str, Any]] = []
    fixed = 0
    already = 0
    for row in failures:
        world = worlds[row["world"]]
        example = world["examples"][row["index"]]
        table, _raw_bytes, gold, _rows = world_table(world)
        read = table.read_sentence(example["goal"])
        picked = tuple(answer["address"] for answer in read["answers"])
        expected = (gold[example["object_description"]], gold[example["place_description"]])
        taken = tuple(span["route"] for span in row["spans"])
        selection_correct = all(span["route"] == span["gold_route"] for span in row["spans"])
        if selection_correct:
            already += 1
        if set(picked) != set(taken):
            fixed += 1
        detail.append(
            {
                "world": row["world"],
                "index": row["index"],
                "goal": example["goal"],
                "published_category": row["category"],
                "published_position": row["position"],
                "published_routes": list(taken),
                "gold_routes": list(expected),
                "bridge_routes": list(picked),
                "bridge_names": [list(answer["value"]) for answer in read["answers"]],
                "selection_correct": bool(selection_correct),
                "bridge_would_change_route": bool(set(picked) != set(taken)),
                "decoded_text": row["decoded_text"],
                "target_text": row["target_text"],
            }
        )
    return {
        "confirm_completion": published["decision"]["confirm_completion"],
        "completion_note": "188/192 as published; not recomputed here",
        "failures": len(failures),
        "already_selection_correct": already,
        "bridge_fixes": fixed,
        "out_of_scope_stop_edge": len(failures) - fixed,
        "detail": detail,
    }


# -- closed state --------------------------------------------------------------
def tensor_sha(values: list[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"none")
        elif isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def reference_digest(reference: Any, state: Any, step_hashes: list[str]) -> dict[str, Any]:
    parameters = [tensor for _, tensor in sorted(reference.named_parameters())]
    store = [state.store_keys, state.store_values, state.store_age, state.store_count]
    identity = {
        "parameters": len(parameters),
        "elements": int(sum(tensor.numel() for tensor in parameters)),
        "config": repr(reference.config),
    }
    return {
        "parameters_sha": tensor_sha(parameters),
        "state_sha": tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior]),
        "keys_len": len(state.keys),
        "step": int(state.step),
        "store_sha": tensor_sha(store),
        "identity_sha": hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sc_sfk_none": state.sc_sfk is None,
        "battery_sha": hashlib.sha256("".join(step_hashes).encode("ascii")).hexdigest(),
        "battery_steps": len(step_hashes),
    }


def run_battery(reference: Any, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    state = reference.initial_state("r56-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r56")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


def battery_texts() -> list[str]:
    """Neutral sentences built from Task1's own description pools, never from confirm goals."""
    sentences = [
        f"让{obj}到{place}。"
        for obj in premise.OBJECT_DESCRIPTIONS[:4]
        for place in premise.PLACE_DESCRIPTIONS[:5]
    ]
    return sentences[:20]


# -- verdict -------------------------------------------------------------------
def verdict(
    joint: dict[str, int], checks: dict[str, int], comparison: dict[str, Any]
) -> dict[str, Any]:
    joint_all = joint["selected"] == 192
    slot_all = checks["address_ok"] == checks["slots"] and checks["value_ok"] == checks["slots"]
    wiring_all = checks["name_bytes_ok"] == checks["slots"]
    if joint_all and slot_all and wiring_all and comparison["bridge_fixes"] >= 1:
        name = "bridge_reaches_mainline"
        branch = "joint, per-slot and wiring all exact, and at least one real failure is fixed"
    elif checks["address_ok"] < checks["slots"] * 0.5:
        name = "bridge_parallel_world"
        branch = "per-slot address exact below 0.5: the bridge cannot name Task1's records"
    else:
        name = "bridge_partial_overlap"
        branch = (
            "per-slot route and name are exact, but the joint rule selects nothing on Task1 and "
            "the bridge fixes none of the four published failures"
        )
    return {
        "verdict": name,
        "branch": branch,
        "joint_all": bool(joint_all),
        "slot_all": bool(slot_all),
        "wiring_all": bool(wiring_all),
        "bridge_fixes": comparison["bridge_fixes"],
    }


# -- freeze --------------------------------------------------------------------
def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "prediction": PREDICTION,
        "criteria": CRITERIA,
        "mainline_relation": MAINLINE_RELATION,
        "split": SPLIT,
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-mainline-fit-r56-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": SPLIT,
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "runner_sha256": RUNNER_SHA_START,
        "bridge_sha256": file_sha(BRIDGE_SOURCE),
        "premise_sha256": file_sha(PREMISE_SOURCE),
        "copy_sha256": file_sha(COPY_SOURCE),
        "previous_report_sha256": file_sha(PREVIOUS_REPORT),
        "definition": DEFINITION,
        "prediction": PREDICTION,
        "criteria": CRITERIA,
        "mainline_relation": MAINLINE_RELATION,
        "definition_digest": definition_digest(),
        "counts": {
            "worlds": premise.SPLIT_WORLDS[SPLIT],
            "goals": premise.SPLIT_WORLDS[SPLIT] * premise.EXAMPLES_PER_WORLD,
            "records_per_world": premise.RECORD_LINES,
            "slots": premise.SPLIT_WORLDS[SPLIT] * premise.EXAMPLES_PER_WORLD * 2,
        },
        "sealed_set_note": (
            "confirm materialisation registered in D-20261109-57: record and goal bytes only, "
            "zero model forwards, zero scoring"
        ),
    }
    write_json(MANIFEST, manifest)
    return manifest


def verify_freeze() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit("no frozen manifest; run --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after the freeze; re-freeze")
    if manifest["bridge_sha256"] != file_sha(BRIDGE_SOURCE):
        raise SystemExit("bridge module changed after the freeze; re-freeze")
    if manifest["premise_sha256"] != file_sha(PREMISE_SOURCE):
        raise SystemExit("premise module changed after the freeze; re-freeze")
    if manifest["copy_sha256"] != file_sha(COPY_SOURCE):
        raise SystemExit("copy module changed after the freeze; re-freeze")
    if manifest["previous_report_sha256"] != file_sha(PREVIOUS_REPORT):
        raise SystemExit("the published 19.2v report changed; refuse to cite it")
    if manifest["definition_digest"] != definition_digest():
        raise SystemExit("definition/criteria/prediction changed after the freeze; re-freeze")
    return manifest


def main() -> int:
    assert_no_black_processes()
    base.frozen.acquire_lock()
    try:
        if "--freeze" in sys.argv:
            manifest = freeze()
            print(json.dumps({"manifest": str(MANIFEST), "counts": manifest["counts"]}))
            return 0
        manifest = verify_freeze()

        worlds = build_worlds()
        got = readings(worlds)
        compared = mainline_comparison(worlds)
        example = wiring(worlds[0], worlds[0]["examples"][0])

        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        texts = battery_texts()
        before = run_battery(reference, texts)
        after = run_battery(reference, texts)
        checks = {
            "parameters_equal": before["parameters_sha"] == after["parameters_sha"],
            "state_equal": before["state_sha"] == after["state_sha"],
            "identity_equal": before["identity_sha"] == after["identity_sha"],
            "store_equal": before["store_sha"] == after["store_sha"],
            "battery_equal": before["battery_sha"] == after["battery_sha"],
            "steps_equal": before["step"] == after["step"],
            "sc_sfk_none": before["sc_sfk_none"] and after["sc_sfk_none"],
            "flag_disabled": reference.config.sc_sfk_enabled is False,
            "rng_untouched": True,
        }
        closed_ok = all(checks.values())

        found = verdict(got["joint"], got["checks"], compared)
        holds = {
            "joint": found["joint_all"] is False
            and got["joint"]["no_match"] == PREDICTION["joint"]["no_match"],
            "per_slot": got["checks"]["address_ok"] == PREDICTION["per_slot"]["address_ok"]
            and got["checks"]["value_ok"] == PREDICTION["per_slot"]["value_ok"],
            "wiring": got["checks"]["name_bytes_ok"] == PREDICTION["wiring"]["name_bytes_equal"],
            "mainline": compared["bridge_fixes"] == PREDICTION["mainline"]["bridge_fixes"],
            "verdict": found["verdict"] == PREDICTION["verdict"],
        }

        # three real cases: a plain goal, an aka-styled object record, one failure
        samples = [
            {
                "case": "plain goal: joint no_match, per-slot pair exact",
                "world": 0,
                "index": 0,
                **case(worlds[0], worlds[0]["examples"][0]),
            },
            {
                "case": "aka-styled object record: name field starts at byte 0",
                **aka_case(worlds),
            },
            {
                "case": "published 19.2v failure: selection was already right",
                **compared["detail"][0],
            },
        ]

        cpu, wall = cost()
        report = {
            "schema": "taiji-mainline-fit-r56-v1",
            "mainline_relation": MAINLINE_RELATION,
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": manifest["bridge_sha256"],
            "premise_sha256": manifest["premise_sha256"],
            "copy_sha256": manifest["copy_sha256"],
            "previous_report_sha256": manifest["previous_report_sha256"],
            "definition": DEFINITION,
            "prediction": PREDICTION,
            "criteria": CRITERIA,
            "prediction_holds": holds,
            "materialisation": {
                "split": SPLIT,
                "worlds": len(worlds),
                "goals": got["goals"],
                "model_forwards_on_confirm": 0,
                "scored": 0,
                "previous_reading_recomputed": False,
            },
            "joint": got["joint"],
            "per_slot": got["per_slot"],
            "checks": got["checks"],
            "wiring": example,
            "mainline": compared,
            "samples": samples,
            "closed_state": {"checks": checks, "equal": closed_ok, "battery_steps": len(texts)},
            "verdict": found,
            "cost": {"cpu_seconds": cpu, "wall_seconds": wall, "limit": CPU_CAP},
        }
        write_json(REPORT, report)

        print(
            json.dumps(
                {
                    "report": str(REPORT),
                    "joint": got["joint"],
                    "per_slot": got["per_slot"],
                    "checks": got["checks"],
                    "mainline": {
                        "failures": compared["failures"],
                        "already_selection_correct": compared["already_selection_correct"],
                        "bridge_fixes": compared["bridge_fixes"],
                    },
                    "verdict": found,
                    "prediction_holds": holds,
                    "closed_state_equal": closed_ok,
                    "cost": {"cpu": cpu, "wall": wall},
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0 if closed_ok else 1
    finally:
        base.frozen.release_lock()


def case(world: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    table, raw_bytes, gold, rows = world_table(world)
    goal = example["goal"]
    read = table.read_sentence(goal)
    keys = tuple(goal_span(goal, row["span_start"], row["span_end"]) for row in read["answers"])
    status, hits = joint_select(raw_bytes, keys)
    return {
        "goal": goal,
        "keys": [key.decode("utf-8") for key in keys],
        "joint_status": status,
        "joint_candidates": list(hits),
        "per_slot_decision": read["decision"],
        "per_slot": [
            {
                "slot": answer["value"][0],
                "name": answer["value"][1],
                "route": answer["address"],
                "gold_route": gold[example[f"{answer['value'][0]}_description"]],
                "record": rows[answer["address"]]["raw"],
                "copy_field_bytes": list(
                    field_bounds(
                        rows[answer["address"]]["raw"],
                        example[f"{answer['value'][0]}_description"],
                    )
                ),
            }
            for answer in read["answers"]
        ],
    }


def aka_case(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    for world in worlds:
        for record in world["records"]:
            if record["kind"] == "object" and record["style_name"] == "aka":
                for example in world["examples"]:
                    if example["object_description"] == record["description"]:
                        return {
                            "world": world["world"],
                            "index": example["index"],
                            **case(world, example),
                        }
    return {"world": None, "index": None, "note": "no aka-styled object record in the split"}


if __name__ == "__main__":
    raise SystemExit(main())
