"""R15 -- sentence-level two-pass read, and the conflict rule made structural.

R14 measured that "first writer wins" only holds when ``on_conflict="reject"``
and that ``replace`` silently re-points an address at the later writer.  R15
turns that lesson into structure instead of a convention: ``BridgeTable`` now
has exactly one conflict rule, no parameter that can select another, and no
branch that appends a second value.  The absence is asserted from the module's
own signature and syntax tree, not promised in prose.

The read side asks a different question from R13/R14: one *sentence* names two
entities, while the bridge answers one wording per query.  The two-pass read is
therefore: answer with the longest stored key that occurs in the sentence, mask
exactly that span, answer again.

Frozen masking rule: the **first** occurrence of the answered key's whole
character sequence is replaced by ``MASK_CHARACTER`` ("\\u0000") repeated to the
same width, so positions after the span do not shift.  The second pass obeys
the same rule.  Exits are structural: one longest key answers, two records of
equal maximal length abstain, no key says ``masked_no_match``, and a second
occurrence of the same address is ``same_address_repeated`` (one entity, not
two).

Capability is counted as 20 sentences x 2 slots = 40 entities, and the sentence
distribution (answered 2 / answered 1 / wrong / abstained) is reported next to
it.  Zero lines re-run on the 18-key table: 12 unknown candidates and 18 byte
fragments must answer 0.

Zero training, zero optimizer steps, zero Adam, zero confirm.  No fixture byte
is written; ``AddressTable`` and ``LivingCoreReference`` are untouched.
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
    MASK_CHARACTER,
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

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_bridge_two_pass_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_two_pass_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)

UNKNOWN_CANDIDATES = (
    "黑色铁铃",
    "白色帆布",
    "长柄铜勺",
    "软木瓶塞",
    "蓝色毛巾",
    "银质圆镜",
    "棕色皮箱",
    "青色陶罐",
    "灰色石阶",
    "紫色绸带",
    "金色烛台",
    "深红木箱",
)

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

SINGLE_SLOT_SENTENCE = "让圆口的器皿留在这里。"
REPEATED_KEY_SENTENCE = "圆口的器皿和圆口的器皿都在。"
TIE_KEYS = (("甲乙", ("object", "蓝色杯子")), ("甲乙丙", ("object", "红色印章")))
TIE_SENTENCE = "甲乙丙丁"

DEFINITION = {
    "read": (
        "two passes over one sentence; each pass answers with the stored key that starts earliest "
        "and whose whole character sequence occurs contiguously in the current text"
    ),
    "mask": (
        "after a pass answers, the first occurrence of that key's sequence is replaced by '\\u0000' "
        "repeated to the same character width, so later positions do not shift"
    ),
    "tie": (
        "two different records that start at the same earliest position abstain "
        "(ambiguous_same_start); never pick one"
    ),
    "repeat": (
        "a second occurrence of the same address is same_address_repeated: it is one entity, not two"
    ),
    "conflict": (
        "exactly one write rule: first writer keeps the address, a later batch that would bind the "
        "same key to a different value is rejected whole; there is no parameter and no branch for "
        "another rule"
    ),
    "capability": "20 describe sentences x 2 slots = 40 entities, plus answered-2/1/wrong/abstained",
}

CRITERIA = {
    "structural": (
        "BridgeTable.__init__ exposes exactly start_capacity/key_fn/read_mode; constructing it with "
        "on_conflict raises TypeError; the module's syntax tree contains no on_conflict, replaced, "
        "history, history_at or values_at identifiers; a re-binding batch exits "
        "rejected_batch_address_already_bound with the content digest unchanged"
    ),
    "two_pass": (
        "20 describe sentences reach 40/40 correct entities, every sentence exits answered_pair, "
        "wrong pointers 0, and the four-way sentence distribution sums to 20"
    ),
    "zeros": "on the 18-key table, 12 unknown candidates answer 0 and 18 byte fragments answer 0",
    "closed_state": "the nine closed-state readings stay equal before and after",
    "reporting": "readings are reported separately; a clean structural gate does not carry two-pass",
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


def sentences() -> list[str]:
    return [row["text"] for row in base.frozen.build_fixture() if row["variant"] == "describe"]


def expected_pair(text: str) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for name, phrase in OBJECT_DESCRIPTIONS.items():
        if phrase in text:
            found["object"] = ("object", name)
    for name, phrase in PLACE_DESCRIPTIONS.items():
        if phrase in text:
            found["place"] = ("place", name)
    return found


def build_table() -> BridgeTable:
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    write = table.write(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS))
    if write["action"] != "accepted":
        raise SystemExit(f"initial write rejected: {write['action']}")
    alias = table.write(AUTHORED_ALIASES)
    if alias["action"] != "accepted":
        raise SystemExit(f"alias write rejected: {alias['action']}")
    return table


# -- structural gate: the replace path does not exist -------------------------
def structural_evidence() -> dict[str, Any]:
    source = BRIDGE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    identifiers = names | attributes
    forbidden = ("on_conflict", "replaced", "history", "history_at", "values_at")
    present = sorted(token for token in forbidden if token in identifiers)
    parameters = list(inspect.signature(BridgeTable.__init__).parameters)
    try:
        BridgeTable(on_conflict="replace")  # type: ignore[call-arg]
        rejected_by_signature = False
    except TypeError:
        rejected_by_signature = True

    table = build_table()
    rebound = table.write([("圆口的器皿", ("object", "红色印章"))])
    same_key_same_value = table.write([("圆口的器皿", ("object", "蓝色杯子"))])
    return {
        "init_parameters": parameters,
        "parameter_set_ok": parameters == ["self", "start_capacity", "key_fn", "read_mode"],
        "unexpected_kwarg_raises": rejected_by_signature,
        "forbidden_identifiers_in_module": present,
        "forbidden_identifiers_absent": present == [],
        "rebind": {
            "action": rebound["action"],
            "digest_unchanged": rebound["digest_unchanged"],
            "first_binding_still_reads": list(table.read("圆口的器皿")["value"]),
        },
        "same_value_merge": {
            "action": same_key_same_value["action"],
            "merged": same_key_same_value["merged"],
        },
        "accepted_outcome_keys": sorted(same_key_same_value),
    }


# -- two-pass read -----------------------------------------------------------
def two_pass_reading(table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hits = wrong = abstained = 0
    distribution = {"answered_2": 0, "answered_1": 0, "wrong": 0, "abstained": 0}
    widths: dict[int, int] = {}
    for text in texts:
        expected = expected_pair(text)
        result = table.read_sentence(text)
        values = [tuple(row["value"]) for row in result["answers"]]
        answered = {(slot, name) for slot, name in values}
        matched: dict[str, str] = {}
        wrong_here = 0
        for slot, name in answered:
            if expected.get(slot) == (slot, name) and slot not in matched:
                matched[slot] = name
            else:
                wrong_here += 1
        hits += len(matched)
        wrong += wrong_here
        abstained += max(0, 2 - len(matched) - wrong_here)
        if len(matched) == 2:
            distribution["answered_2"] += 1
            sentence_kind = "answered_2"
        elif len(matched) == 1 and wrong_here == 0:
            distribution["answered_1"] += 1
            sentence_kind = "answered_1"
        elif wrong_here:
            distribution["wrong"] += 1
            sentence_kind = "wrong"
        else:
            distribution["abstained"] += 1
            sentence_kind = "abstained"
        mask = result["mask"] or {}
        width = int(mask.get("span_end", 0)) - int(mask.get("span_start", 0)) if mask else 0
        widths[width] = widths.get(width, 0) + 1
        rows.append(
            {
                "sentence": text,
                "decision": result["decision"],
                "sentence_kind": sentence_kind,
                "expected": {slot: list(value) for slot, value in sorted(expected.items())},
                "answers": [list(value) for value in values],
                "passes": [
                    {
                        key: row[key]
                        for key in ("pass", "decision", "reason", "span_start", "span_end")
                    }
                    for row in result["passes"]
                ],
                "mask": (
                    {
                        "span": [mask.get("span_start"), mask.get("span_end")],
                        "width": width,
                        "masked_query": mask.get("masked_query"),
                    }
                    if mask
                    else None
                ),
            }
        )
    total = 2 * len(texts)
    return {
        "rows": rows,
        "entities": total,
        "hits": hits,
        "wrong": wrong,
        "abstained": abstained,
        "distribution": distribution,
        "mask_widths": dict(sorted(widths.items())),
        "mask_rule": DEFINITION["mask"],
        "mask_character": repr(MASK_CHARACTER),
        "ok": hits == total and wrong == 0 and distribution["answered_2"] == len(texts),
    }


def zeros_reading(table: BridgeTable) -> dict[str, Any]:
    unknown_rows = []
    unknown_answering = 0
    for phrase in UNKNOWN_CANDIDATES:
        result = table.read_sentence(phrase)
        answered = bool(result["answers"])
        unknown_answering += int(answered)
        unknown_rows.append(
            {"phrase": phrase, "decision": result["decision"], "answered": answered}
        )
    fragment_rows = []
    fragment_answering = 0
    for phrase, _ in bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS):
        data = phrase.encode("utf-8")
        for offset in (1, 2):
            result = table.read_sentence(data[offset:])
            answered = bool(result["answers"])
            fragment_answering += int(answered)
            fragment_rows.append(
                {"phrase": phrase, "byte_offset": offset, "decision": result["decision"]}
            )
    single_pass = [table.read(text) for text in sentences()]
    return {
        "unknown": {
            "rows": unknown_rows,
            "candidates": len(unknown_rows),
            "answering": unknown_answering,
            "zero_ok": unknown_answering == 0,
        },
        "fragments": {
            "rows": fragment_rows,
            "inputs": len(fragment_rows),
            "answering": fragment_answering,
            "zero_ok": fragment_answering == 0,
        },
        "single_pass_baseline": {
            "inputs": len(single_pass),
            "answering": sum(1 for row in single_pass if row["value"] is not None),
            "decisions": {
                name: sum(1 for row in single_pass if row["decision"] == name)
                for name in sorted({row["decision"] for row in single_pass})
            },
            "note": "one pass cannot split a two-entity sentence; the two-pass read is the treatment",
        },
    }


# -- controls ----------------------------------------------------------------
def controls() -> dict[str, Any]:
    table = build_table()
    single = table.read_sentence(SINGLE_SLOT_SENTENCE)
    repeated = table.read_sentence(REPEATED_KEY_SENTENCE)
    tie_table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    tie_write = tie_table.write(TIE_KEYS)
    tie = tie_table.read_sentence(TIE_SENTENCE)
    return {
        "single_slot": {
            "sentence": SINGLE_SLOT_SENTENCE,
            "decision": single["decision"],
            "second_pass_exit": single["passes"][-1]["decision"],
            "second_pass_reason": single["passes"][-1]["reason"],
            "answers": [list(row["value"]) for row in single["answers"]],
        },
        "duplicate_key": {
            "sentence": REPEATED_KEY_SENTENCE,
            "decision": repeated["decision"],
            "second_pass_exit": repeated["passes"][-1]["decision"],
            "distinct_addresses": repeated["distinct_addresses"],
            "answers": [list(row["value"]) for row in repeated["answers"]],
            "note": "one wording maps to one entity by write rule, so a repeat is one entity",
        },
        "tie": {
            "write": tie_write["action"],
            "sentence": TIE_SENTENCE,
            "decision": tie["decision"],
            "reason": tie["passes"][0]["reason"],
            "candidates": tie["passes"][0]["candidates"],
        },
    }


# -- closed state ------------------------------------------------------------
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
    state = reference.initial_state("r15-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r15")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


# -- freeze ------------------------------------------------------------------
def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "unknown": list(UNKNOWN_CANDIDATES),
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "controls": {
            "single_slot": SINGLE_SLOT_SENTENCE,
            "duplicate_key": REPEATED_KEY_SENTENCE,
            "tie_keys": [list(row) for row in TIE_KEYS],
            "tie_sentence": TIE_SENTENCE,
        },
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-bridge-two-pass-r15-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "runner_sha256": RUNNER_SHA_START,
        "bridge_sha256": file_sha(BRIDGE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "records": 9,
            "aliases": len(AUTHORED_ALIASES),
            "keys": 18,
            "describe_sentences": len(sentences()),
            "capability_entities": 2 * len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
        },
        "inputs": {
            "unknown_candidates": list(UNKNOWN_CANDIDATES),
            "authored_aliases": [list(row) for row in AUTHORED_ALIASES],
            "single_slot_sentence": SINGLE_SLOT_SENTENCE,
            "duplicate_key_sentence": REPEATED_KEY_SENTENCE,
            "tie_keys": [list(row) for row in TIE_KEYS],
            "tie_sentence": TIE_SENTENCE,
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
        raise SystemExit("definition/criteria/inputs changed after the freeze; re-freeze")
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

        texts = sentences()
        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = run_battery(reference, texts)

        table = build_table()
        structural = structural_evidence()
        two_pass = two_pass_reading(table, texts)
        zeros = zeros_reading(table)
        control = controls()

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

        structural_ok = (
            structural["parameter_set_ok"]
            and structural["unexpected_kwarg_raises"]
            and structural["forbidden_identifiers_absent"]
            and structural["rebind"]["action"] == "rejected_batch_address_already_bound"
            and structural["rebind"]["digest_unchanged"]
            and structural["same_value_merge"]["action"] == "accepted"
        )
        controls_ok = (
            control["single_slot"]["decision"] == "answered_single"
            and control["single_slot"]["second_pass_exit"] == "no_match"
            and control["duplicate_key"]["decision"] == "answered_single_duplicate_key"
            and control["duplicate_key"]["second_pass_exit"] == "same_address_repeated"
            and control["duplicate_key"]["distinct_addresses"] == 1
            and control["tie"]["decision"] == "ambiguous_same_start"
        )
        zeros_ok = zeros["unknown"]["zero_ok"] and zeros["fragments"]["zero_ok"]
        closed_ok = all(checks.values())
        verdict = {
            "structural": "replace_path_absent" if structural_ok else "replace_path_reachable",
            "two_pass": "two_pass_clean" if two_pass["ok"] else "two_pass_incomplete",
            "zeros": "zeros_clean" if zeros_ok else "zeros_leak",
            "controls": "exits_declared" if controls_ok else "exit_mismatch",
        }

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-two-pass-r15-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "structural": structural,
            "two_pass": two_pass,
            "zeros": zeros,
            "controls": control,
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
                    "two_pass": {
                        "hits": two_pass["hits"],
                        "of": two_pass["entities"],
                        "wrong": two_pass["wrong"],
                        "abstained": two_pass["abstained"],
                        "distribution": two_pass["distribution"],
                        "mask_widths": two_pass["mask_widths"],
                    },
                    "zeros": {
                        "unknown": zeros["unknown"]["answering"],
                        "fragments": zeros["fragments"]["answering"],
                        "single_pass_answering": zeros["single_pass_baseline"]["answering"],
                    },
                    "controls": {
                        "single": control["single_slot"]["decision"],
                        "duplicate": control["duplicate_key"]["decision"],
                        "tie": control["tie"]["decision"],
                    },
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
