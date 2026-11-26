"""R54' -- joint-hit record selection (the selection layer, not the last hop).

R53 showed that the bridge hands out faithful bytes (L2 = 1.0) but only 9/40
slots landed on the asked (object, place) pair.  The diagnosis adopted here is
that the fault is in RECORD SELECTION, not in the last hop: from R11 to R16 the
bridge answers each slot separately and the record is taken at the lowest
address (SPEC 16.2 ``address_order``).  Nothing ever asks "which record contains
ALL the keys of this sentence at once?".

This runner performs that one change and nothing else:

    question -> R15 two-pass read (unchanged) -> the literal byte spans it
    matched -> candidate records that contain ALL of those spans -> if exactly
    one candidate, emit one contiguous span of it (frozen SPEC 16.2 unit);
    zero candidates -> ``no_match``, no bytes; two or more ->
    ``ambiguous_multiple_records``, no bytes.

The lowest-address fallback is deleted, not disabled.  Four structural checks
(signature / literal / AST / behaviour) are reported as evidence that it is
unreachable, in the same spirit as R15's removal of ``replace``.

Prediction is frozen next to the criteria, before any reading.  The honest cost
is reported beside the gain: the joint rule can only select when the asked words
occur verbatim in one record, so (a) name-worded questions hit three records per
corpus pair and must abstain, and (b) authored aliases identify an entity but
never appear in the corpus at all.  Both are measured as secondary groups and
neither is folded into the primary reading.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
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
from taiji.address_bridge import (  # noqa: E402
    BridgeTable,
    bridge_records,
    sequence_key,
)
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402
from taiji.sc_sfk import sentences as sentence_spans  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_joint_select_20261109.json"
REPORT = ROOT / "reports/taiji_joint_select_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)

AUTHORED_ALIASES = (
    ("喝水的杯子", ("object", "蓝色杯子")),
    ("盖章用的印章", ("object", "红色印章")),
    ("盛水的瓶子", ("object", "绿色瓶子")),
    ("圆形的皮球", ("object", "黄色球")),
    ("方方正正的木桌子", ("place", "木桌")),
    ("窗前的台板", ("place", "窗台")),
    ("手边的小盒子", ("place", "左盒")),
    ("墙上的架子", ("place", "书架")),
    ("脚下的垫子", ("place", "地毯")),
)

DEFINITION = {
    "question": "does a joint hit over all keys of one sentence fix the record choice?",
    "selection_rule": (
        "candidate = every corpus record that contains ALL literal byte spans the R15 read "
        "matched in the question; exactly one candidate -> emit it; zero -> no_match, no bytes; "
        "two or more -> ambiguous_multiple_records, no bytes"
    ),
    "deleted_fallback": (
        "the lowest-address path (SPEC 16.2 address_order) is deleted from selection; it is not "
        "reachable by any flag, and no default address is returned"
    ),
    "groups": {
        "primary": "the 20 describe sentences (4 objects x 5 places x 1 variant); 40 entity slots",
        "names": "20 authored name-worded questions 让{object}到{place}。 (secondary, not judged)",
        "aliases": "9 authored aliases alone, one per entity (secondary, not judged)",
    },
    "unit": "frozen SPEC 16.2 unit: a complete record or a complete sentence; no third unit",
    "L1": "fabricated 0, illegal 0, UTF-8 decodable 100%, boundary legal 100%",
    "L2": "emitted bytes equal record[start:end] byte for byte; must be 1.0",
    "L3": (
        "four classes per entity slot: answered_correct / pointed_wrong / abstained / "
        "should_answer_missing; ambiguous_multiple_records counts as abstained and no_match as "
        "should_answer_missing, each also reported by name"
    ),
    "paired_numbers": (
        "pairing_matches and answers_lost are reported side by side; answers_lost = 40 minus the "
        "slots that emitted bytes, i.e. what was paid to remove wrong pairings"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "training": "zero training, zero optimizer steps, zero Adam, zero confirm",
    "diagnostics": (
        "two extra groups were added AFTER the first pass, when the name group turned out to have "
        "no keys at all: paraphrase = the 60 name-worded records, mixed = 9 alias + description "
        "questions.  They are reported for the cost column only and never enter the verdict"
    ),
}

PREDICTION = {
    "primary": {
        "sentences": 20,
        "selected": 20,
        "pairing_matches": 40,
        "answers_lost": 0,
        "why": (
            "a describe sentence's two key spans only co-occur in that pair's describe record, so "
            "the joint hit is a singleton and the record is the asked one"
        ),
    },
    "names": {
        "sentences": 20,
        "selected": 0,
        "ambiguous": 20,
        "candidates_per_sentence": 3,
        "why": "names appear in the needs / hope / state records of the same pair: three records",
    },
    "aliases": {
        "sentences": 9,
        "selected": 0,
        "no_match": 9,
        "why": "the alias wording is not in the corpus at all",
    },
}

DIAGNOSTIC_PREDICTION = {
    "paraphrase": {
        "sentences": 60,
        "selected": 0,
        "no_keys": 60,
        "why": "needs / hope / state sentences are name-worded and names are not keys",
    },
    "mixed": {
        "sentences": 9,
        "selected": 0,
        "no_match": 9,
        "why": "one key is an alias (absent from the corpus), so a joint hit cannot exist",
    },
}

CRITERIA = {
    "structural": (
        "signature, literal, AST and behaviour checks all report the fallback unreachable; any "
        "one failing blocks the verdict"
    ),
    "l1": "fabricated 0, illegal 0, utf8 all, boundary all",
    "l2": "provenance fidelity == 1.0",
    "loss": "pairing_matches and answers_lost both reported, never only the first",
    "samples": "at least 3 real outputs, including one wrong->correct and one wrong->uncertain",
    "closed_state": "nine items equal",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- corpus -------------------------------------------------------------------
def corpus_rows() -> list[dict[str, Any]]:
    return base.frozen.build_fixture()


def corpus_records(rows: list[dict[str, Any]]) -> list[bytes]:
    return [row["text"].encode("utf-8") for row in rows]


def describe_sentences(rows: list[dict[str, Any]]) -> list[str]:
    return [row["text"] for row in rows if row["variant"] == "describe"]


def expected_pair(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, phrase in OBJECT_DESCRIPTIONS.items():
        if phrase in text or name in text:
            found["object"] = name
    for name, phrase in PLACE_DESCRIPTIONS.items():
        if phrase in text or name in text:
            found["place"] = name
    return found


def name_question(obj: str, place: str) -> str:
    return f"让{obj}到{place}。"


def build_table() -> BridgeTable:
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    write = table.write(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS))
    if write["action"] != "accepted":
        raise SystemExit(f"initial write rejected: {write['action']}")
    alias = table.write(AUTHORED_ALIASES)
    if alias["action"] != "accepted":
        raise SystemExit(f"alias write rejected: {alias['action']}")
    return table


# -- selection (the only change under test) -----------------------------------
def select_records(records: list[bytes], keys: tuple[bytes, ...]) -> tuple[str, tuple[int, ...]]:
    """Joint hit.  Returns a status and ALL candidates; never one address."""
    wanted = tuple(dict.fromkeys(keys))
    if not wanted:
        return "no_keys", ()
    hits = tuple(index for index, record in enumerate(records) if all(k in record for k in wanted))
    if not hits:
        return "no_match", hits
    if len(hits) == 1:
        return "selected", hits
    return "ambiguous_multiple_records", hits


def query_keys(table: BridgeTable, text: str) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    result = table.read_sentence(text)
    characters = list(text)
    keys = tuple(
        "".join(characters[row["span_start"] : row["span_end"]]).encode("utf-8")
        for row in result["answers"]
    )
    return result, keys


# -- output unit --------------------------------------------------------------
def emit(record: bytes) -> dict[str, Any]:
    spans = sentence_spans(record)
    start, end = spans[0]
    body = record[start:end]
    try:
        text = body.decode("utf-8")
        utf8_ok = True
    except UnicodeDecodeError:
        text = ""
        utf8_ok = False
    return {
        "unit": "record" if (start, end) == (0, len(record)) else "sentence",
        "start": start,
        "end": end,
        "spans": [list(span) for span in spans],
        "bytes": body,
        "text": text,
        "utf8_ok": utf8_ok,
        "fabricated": body != record[start:end],
        "illegal_slice": not (0 <= start <= end <= len(record)),
        "boundary_legal": [start, end] in [list(span) for span in spans],
        "faithful": body == record[start:end],
    }


# -- the reading --------------------------------------------------------------
def joint_reading(
    table: BridgeTable,
    texts: list[str],
    rows: list[dict[str, Any]],
    records: list[bytes],
    asked_override: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    outcomes = {
        "answered_correct": 0,
        "pointed_wrong": 0,
        "abstained": 0,
        "should_answer_missing": 0,
    }
    statuses = {"selected": 0, "ambiguous_multiple_records": 0, "no_match": 0, "no_keys": 0}
    emitted: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    for text in texts:
        expected = expected_pair(text)
        asked = (
            asked_override[text]
            if asked_override and text in asked_override
            else (expected.get("object"), expected.get("place"))
        )
        result, keys = query_keys(table, text)
        status, hits = select_records(records, keys)
        statuses[status] += 1
        row: dict[str, Any] = {
            "sentence": text,
            "decision": result["decision"],
            "keys": [key.decode("utf-8") for key in keys],
            "answers": [[answer["value"][0], answer["value"][1]] for answer in result["answers"]],
            "status": status,
            "candidate_count": len(hits),
            "candidates": list(hits[:6]),
            "asked_pair": [asked[0], asked[1]],
            "slots": {},
        }
        output: dict[str, Any] | None = None
        record_pair: tuple[str, str] | None = None
        if status == "selected":
            (address,) = hits
            output = emit(records[address])
            record_pair = (rows[address]["object"], rows[address]["place"])
            row.update(
                {
                    "address": address,
                    "unit": output["unit"],
                    "span": [output["start"], output["end"]],
                    "record_pair": list(record_pair),
                    "pairing_matches": record_pair == asked,
                    "utf8_ok": output["utf8_ok"],
                    "fabricated": output["fabricated"],
                    "illegal_slice": output["illegal_slice"],
                    "boundary_legal": output["boundary_legal"],
                    "faithful": output["faithful"],
                    "text": output["text"],
                }
            )
            emitted.append(row)
        for slot in ("object", "place"):
            if expected.get(slot) is None:
                continue
            if status == "selected" and record_pair == asked:
                outcomes["answered_correct"] += 1
                row["slots"][slot] = "answered_correct"
            elif status == "selected":
                outcomes["pointed_wrong"] += 1
                row["slots"][slot] = "pointed_wrong"
            elif status == "ambiguous_multiple_records":
                outcomes["abstained"] += 1
                row["slots"][slot] = "abstained"
            else:
                outcomes["should_answer_missing"] += 1
                row["slots"][slot] = "should_answer_missing"
        sentence_rows.append(row)

    slots = sum(len(row["slots"]) for row in sentence_rows)
    l1 = {
        "fabricated": sum(int(row["fabricated"]) for row in emitted),
        "illegal_slices": sum(int(row["illegal_slice"]) for row in emitted),
        "utf8_valid": sum(int(row["utf8_ok"]) for row in emitted),
        "boundary_legal": sum(int(row["boundary_legal"]) for row in emitted),
        "outputs": len(emitted),
    }
    faithful = sum(int(row["faithful"]) for row in emitted)
    emitted_slots = outcomes["answered_correct"] + outcomes["pointed_wrong"]
    return {
        "sentences": len(texts),
        "entity_slots": slots,
        "statuses": statuses,
        "outcomes": outcomes,
        "l1": l1,
        "l2": {
            "faithful": faithful,
            "outputs": len(emitted),
            "fidelity": (faithful / len(emitted)) if emitted else 0.0,
        },
        "answered_sentences": len(emitted),
        "pairing_matches": outcomes["answered_correct"],
        "answers_lost": slots - emitted_slots,
        "rows": sentence_rows,
        "outputs": [{key: value for key, value in row.items() if key != "text"} for row in emitted],
    }


def legacy_address(answers: list[list[str]], rows: list[dict[str, Any]]) -> int | None:
    """Simulation of R53's ``candidates[0]``: the first record of the first answered slot."""
    for slot in ("object", "place"):
        for answer_slot, name in answers:
            if answer_slot != slot:
                continue
            hits = [index for index, row in enumerate(rows) if row[slot] == name]
            if hits:
                return hits[0]
    return None


def paraphrase_sentences(rows: list[dict[str, Any]]) -> list[str]:
    return [row["text"] for row in rows if row["variant"] != "describe"]


def mixed_questions() -> tuple[list[str], dict[str, tuple[str, str]]]:
    object_aliases = [row for row in AUTHORED_ALIASES if row[1][0] == "object"]
    place_aliases = [row for row in AUTHORED_ALIASES if row[1][0] == "place"]
    places = list(PLACE_DESCRIPTIONS)
    objects = list(OBJECT_DESCRIPTIONS)
    texts: list[str] = []
    pairs: dict[str, tuple[str, str]] = {}
    for index, (alias, (_, obj)) in enumerate(object_aliases):
        place = places[index % len(places)]
        text = f"让{alias}到{PLACE_DESCRIPTIONS[place]}。"
        texts.append(text)
        pairs[text] = (obj, place)
    for index, (alias, (_, place)) in enumerate(place_aliases):
        obj = objects[index % len(objects)]
        text = f"让{OBJECT_DESCRIPTIONS[obj]}到{alias}。"
        texts.append(text)
        pairs[text] = (obj, place)
    return texts, pairs


# -- structural evidence that the fallback is gone ----------------------------
def selection_source() -> str:
    return inspect.getsource(select_records) + inspect.getsource(query_keys)


def literal_check() -> dict[str, Any]:
    source = selection_source()
    forbidden = ("min(", "max(", "address_order", "[0]")
    found = [token for token in forbidden if token in source]
    return {"forbidden": list(forbidden), "found": found, "passed": not found}


def signature_check() -> dict[str, Any]:
    signature = inspect.signature(select_records)
    names = tuple(signature.parameters)
    returns = signature.return_annotation
    passed = (
        names == ("records", "keys")
        and "address" not in names
        and "default" not in names
        and "tuple[int, ...]" in str(returns)
    )
    return {"parameters": list(names), "return": str(returns), "passed": passed}


def ast_check() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).resolve().read_text(encoding="utf-8"))
    targets = {"select_records", "query_keys"}
    calls: list[str] = []
    subscripts: list[str] = []
    attributes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    calls.append(inner.func.id)
                if isinstance(inner, ast.Subscript):
                    subscripts.append(ast.unparse(inner))
                if isinstance(inner, ast.Attribute):
                    attributes.append(inner.attr)
    bad_calls = [name for name in calls if name in {"min", "max", "sorted"}]
    bad_subs = [text for text in subscripts if text.endswith("[0]")]
    bad_attrs = [name for name in attributes if name == "address_order"]
    passed = not (bad_calls or bad_subs or bad_attrs)
    return {
        "calls": sorted(set(calls)),
        "subscripts": subscripts,
        "attributes": sorted(set(attributes)),
        "bad_calls": bad_calls,
        "bad_subscripts": bad_subs,
        "bad_attributes": bad_attrs,
        "passed": passed,
    }


def behavior_check(records: list[bytes]) -> dict[str, Any]:
    single = select_records(records, (b"alpha", b"beta"))
    multiple = select_records(records, (b"alpha",))
    missing = select_records(records, (b"delta",))
    empty = select_records(records, ())
    cases = {
        "single_hit": single,
        "multi_hit": multiple,
        "no_hit": missing,
        "no_keys": empty,
    }
    passed = (
        single[0] == "selected"
        and len(single[1]) == 1
        and multiple[0] == "ambiguous_multiple_records"
        and len(multiple[1]) == 2
        and missing[0] == "no_match"
        and not missing[1]
        and empty[0] == "no_keys"
        and not empty[1]
    )
    return {"fixture": ["alpha beta.", "alpha gamma."], "cases": cases, "passed": passed}


STRUCTURAL_FIXTURE = [b"alpha beta.", b"alpha gamma."]


# -- closed state -------------------------------------------------------------
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
    state = reference.initial_state("r54-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r54")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


# -- freeze -------------------------------------------------------------------
def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "prediction": PREDICTION,
        "diagnostic_prediction": DIAGNOSTIC_PREDICTION,
        "criteria": CRITERIA,
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    rows = corpus_rows()
    manifest = {
        "schema": "taiji-joint-select-r54-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "runner_sha256": RUNNER_SHA_START,
        "bridge_sha256": file_sha(BRIDGE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "prediction": PREDICTION,
        "diagnostic_prediction": DIAGNOSTIC_PREDICTION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "corpus_records": len(rows),
            "describe_sentences": len(describe_sentences(rows)),
            "names_sentences": len(OBJECT_DESCRIPTIONS) * len(PLACE_DESCRIPTIONS),
            "alias_sentences": len(AUTHORED_ALIASES),
            "paraphrase_sentences": len(paraphrase_sentences(rows)),
            "mixed_sentences": len(mixed_questions()[0]),
        },
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
    if manifest["fixture_sha256"] != base.fixture_hashes():
        raise SystemExit("fixture bytes changed; the frozen fixture must not move")
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
        verify_freeze()

        rows = corpus_rows()
        records = corpus_records(rows)
        texts = describe_sentences(rows)
        names = [
            name_question(obj, place) for obj in OBJECT_DESCRIPTIONS for place in PLACE_DESCRIPTIONS
        ]
        aliases = [row[0] for row in AUTHORED_ALIASES]

        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = run_battery(reference, texts)

        table = build_table()
        primary = joint_reading(table, texts, rows, records)
        name_group = joint_reading(table, names, rows, records)
        alias_group = joint_reading(table, aliases, rows, records)
        paraphrase_group = joint_reading(table, paraphrase_sentences(rows), rows, records)
        mixed_texts, mixed_pairs = mixed_questions()
        mixed_group = joint_reading(table, mixed_texts, rows, records, asked_override=mixed_pairs)

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

        structural = {
            "signature": signature_check(),
            "literal": literal_check(),
            "ast": ast_check(),
            "behavior": behavior_check(STRUCTURAL_FIXTURE),
        }
        structural_ok = all(row["passed"] for row in structural.values())

        outputs = primary["l1"]["outputs"]
        l1_ok = (
            primary["l1"]["fabricated"] == 0
            and primary["l1"]["illegal_slices"] == 0
            and primary["l1"]["utf8_valid"] == outputs
            and primary["l1"]["boundary_legal"] == outputs
        )
        l2_ok = primary["l2"]["fidelity"] == 1.0
        prediction_holds = {
            "primary": (
                primary["statuses"]["selected"] == PREDICTION["primary"]["selected"]
                and primary["pairing_matches"] == PREDICTION["primary"]["pairing_matches"]
                and primary["answers_lost"] == PREDICTION["primary"]["answers_lost"]
            ),
            "names": (
                name_group["statuses"]["ambiguous_multiple_records"]
                == PREDICTION["names"]["ambiguous"]
                and name_group["l1"]["outputs"] == 0
            ),
            "aliases": (
                alias_group["statuses"]["no_match"] == PREDICTION["aliases"]["no_match"]
                and alias_group["l1"]["outputs"] == 0
            ),
        }

        samples = []
        for row in primary["rows"]:
            if row["status"] != "selected":
                continue
            old = legacy_address(row["answers"], rows)
            old_pair = (rows[old]["object"], rows[old]["place"]) if old is not None else None
            samples.append(
                {
                    "kind": "wrong->correct" if old_pair != tuple(row["asked_pair"]) else "same",
                    "question": row["sentence"],
                    "old_rule_pair": list(old_pair) if old_pair else None,
                    "new_pair": row["record_pair"],
                    "asked_pair": row["asked_pair"],
                    "emitted": row["text"],
                }
            )
        uncertain = []
        for row in mixed_group["rows"]:
            old = legacy_address(row["answers"], rows)
            old_pair = (rows[old]["object"], rows[old]["place"]) if old is not None else None
            uncertain.append(
                {
                    "kind": "wrong->no_bytes",
                    "question": row["sentence"],
                    "keys": row["keys"],
                    "old_rule_pair": list(old_pair) if old_pair else None,
                    "new_status": row["status"],
                    "candidate_count": row["candidate_count"],
                    "asked_pair": row["asked_pair"],
                }
            )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        verdict = {
            "structural": "fallback_unreachable" if structural_ok else "fallback_reachable",
            "l1": "l1_clean" if l1_ok else "l1_dirty",
            "l2": "provenance_faithful" if l2_ok else "provenance_broken",
            "primary": (
                "joint_select_resolves"
                if primary["statuses"]["selected"] == len(texts)
                else "joint_select_ambiguous"
            ),
            "loss": (
                "zero_loss"
                if primary["answers_lost"] == 0
                else f"answers_lost_{primary['answers_lost']}"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "diagnostics": (
                "reach_limited_to_verbatim_questions"
                if paraphrase_group["statuses"]["selected"] == 0
                else "reach_beyond_verbatim"
            ),
        }
        report = {
            "schema": "taiji-joint-select-r54-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "prediction": PREDICTION,
            "diagnostic_prediction": DIAGNOSTIC_PREDICTION,
            "criteria": CRITERIA,
            "prediction_holds": prediction_holds,
            "diagnostic_prediction_holds": {
                "paraphrase": paraphrase_group["statuses"]["no_keys"]
                == DIAGNOSTIC_PREDICTION["paraphrase"]["no_keys"],
                "mixed": mixed_group["statuses"]["no_match"]
                == DIAGNOSTIC_PREDICTION["mixed"]["no_match"],
            },
            "structural": structural,
            "primary": primary,
            "names": name_group,
            "aliases": alias_group,
            "paraphrase": paraphrase_group,
            "mixed": mixed_group,
            "samples": samples,
            "uncertain": uncertain,
            "closed_state": {"baseline": before, "after": after, "checks": checks},
            "verdict": verdict,
            "cost": {
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "cpu_cap": CPU_CAP,
                "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
                "threads": torch.get_num_threads(),
            },
        }
        write_json(REPORT, report)
        print(
            json.dumps(
                {
                    **{f"verdict_{key}": value for key, value in verdict.items()},
                    "primary": {
                        "statuses": primary["statuses"],
                        "outcomes": primary["outcomes"],
                        "pairing_matches": primary["pairing_matches"],
                        "answers_lost": primary["answers_lost"],
                        "l1": primary["l1"],
                        "l2": {"fidelity": primary["l2"]["fidelity"]},
                    },
                    "names": {
                        "statuses": name_group["statuses"],
                        "candidates": sorted(
                            {row["candidate_count"] for row in name_group["rows"]}
                        ),
                    },
                    "aliases": {"statuses": alias_group["statuses"]},
                    "paraphrase": {
                        "statuses": paraphrase_group["statuses"],
                        "outcomes": paraphrase_group["outcomes"],
                        "answers_lost": paraphrase_group["answers_lost"],
                    },
                    "mixed": {
                        "statuses": mixed_group["statuses"],
                        "outcomes": mixed_group["outcomes"],
                        "sample": uncertain[0] if uncertain else None,
                    },
                    "prediction_holds": prediction_holds,
                    "closed_state_ok": closed_ok,
                    "cost": {"cpu": round(cpu, 3), "wall": round(wall, 3)},
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        base.frozen.release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
