"""10.21.1: read-only preflight for the composition candidate.

Two questions, no training, no parameter changes:

* can the current 09.8 child, under the *new* 64-byte protocol, complete the six
  template actions at all (real vs domain-permuted records vs no records)?
* does the subject's own descriptor query pick the right record out of the glossary?

The instrument is checked first on an offline oracle and fixed bad policies. If that
fails, the model readings are not claimable: an invalid ruler is not evidence about
the candidate. Each generation example runs on its own isolated zero-H/F copy of the
same frozen parameters; the restored bundle state is used only for the descriptor
reads, exactly as 10.21a did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest, load_candidate, parameter_digest  # noqa: E402
from taiji.composition_read_probe import (  # noqa: E402
    TIE,
    descriptor_read,
)
from taiji.core_09_bundle import BUNDLE_RELATIVE_PATH as CORE_09_BUNDLE_PATH  # noqa: E402
from taiji.core_09_bundle import EXPECTED_BYTES as CORE_09_BUNDLE_BYTES  # noqa: E402
from taiji.core_09_bundle import EXPECTED_SHA256 as CORE_09_BUNDLE_SHA  # noqa: E402
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.native_goal_attributes import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
    action_sentence,
    describe_goal,
)
from taiji.native_goal_glossary import glossary_records  # noqa: E402
from taiji.state_migration import resume_subject  # noqa: E402

BUNDLE = os.environ.get("TAIJI_BUNDLE_PATH", CORE_09_BUNDLE_PATH)
CLOSEOUT = "reports/taiji_core_09_closeout_20261101.json"
EXPECTED_CANDIDATE = "feb0cbcb00e2e16bb66b77db530ae566ce40eecaf2e14852b7cc1661292139dd"
EXPECTED_BUNDLE_SHA = CORE_09_BUNDLE_SHA
EXPECTED_BUNDLE_BYTES = CORE_09_BUNDLE_BYTES

BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200
MAX_NEW = 64
MAX_CALLS = 65
PASS_HITS = 5
ABSENT_HITS = 2
READ_PASS = 4
READ_ABSENT = 1
PROBABILITY_DECIMALS = 8

CONDITIONS = ("real", "none", "permuted")
READ_CONDITIONS = ("real", "permuted", "reversed")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def descriptors() -> list[str]:
    return [*OBJECT_DESCRIPTIONS.values(), *PLACE_DESCRIPTIONS.values()]


def targets() -> list[tuple[str, str, str, str]]:
    return [
        (entity, text, place, place_text)
        for (entity, text) in OBJECT_DESCRIPTIONS.items()
        for (place, place_text) in PLACE_DESCRIPTIONS.items()
    ]


def real_pairs() -> list[tuple[str, str]]:
    return [
        *[(text, entity) for entity, text in OBJECT_DESCRIPTIONS.items()],
        *[(text, entity) for entity, text in PLACE_DESCRIPTIONS.items()],
    ]


def permuted_pairs() -> list[tuple[str, str]]:
    """Domain-internal bijections: swap the two objects, cycle the three places."""
    object_items = [(text, entity) for entity, text in OBJECT_DESCRIPTIONS.items()]
    place_items = [(text, entity) for entity, text in PLACE_DESCRIPTIONS.items()]
    object_entities = [entity for _, entity in object_items]
    place_entities = [entity for _, entity in place_items]
    swapped = [
        (text, object_entities[(index + 1) % len(object_entities)])
        for index, (text, _) in enumerate(object_items)
    ]
    cycled = [
        (text, place_entities[(index + 1) % len(place_entities)])
        for index, (text, _) in enumerate(place_items)
    ]
    return [*swapped, *cycled]


def records_from_pairs(pairs: list[tuple[str, str]]) -> tuple[RawSource, ...]:
    return tuple(
        RawSource(source_id=f"glossary-{index}", raw=f"{text}是{entity}。".encode())
        for index, (text, entity) in enumerate(pairs)
    )


def mapping_of(pairs: list[tuple[str, str]]) -> dict[str, str]:
    return {text: entity for text, entity in pairs}


def reverse_records(records: tuple[RawSource, ...]) -> tuple[RawSource, ...]:
    return tuple(
        RawSource(source_id=record.source_id, raw=record.raw) for record in reversed(records)
    )


def records_sha(records: tuple[RawSource, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.source_id.encode())
        digest.update(record.raw)
    return digest.hexdigest()


def normalize_action(text: str) -> str:
    return text.strip().rstrip("。．.!！?？;；,，、 \t\r\n")


def score_action(raw: bytes, termination: str, expected: str) -> dict:
    try:
        text = raw.decode("utf-8")
        valid = True
    except UnicodeDecodeError:
        text, valid = None, False
    exact = bool(valid and normalize_action(text) == expected)
    return {
        "utf8_valid": valid,
        "text": text,
        "display_only": raw.decode("utf-8", errors="replace"),
        "action_exact": exact,
        "completed_action_exact": bool(exact and termination == "eos"),
    }


def legacy_hit(display: str, obj: str, place: str) -> bool:
    return display.find(obj) != -1 and display.find(place) > display.find(obj)


def state_digest(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in (("h", state.h), ("fast", state.fast), ("nu_ema", state.nu_ema)):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gates_provider(gate_table):
    def provider(index: int) -> dict:
        item = gate_table[index % len(gate_table)]
        if isinstance(item, dict):
            return dict(item)
        return {"copy": item[0], "generate": item[1]}

    return provider


def generation_decode(
    subject,
    provider,
    prompt: str,
    goal: str,
    records: tuple[RawSource, ...],
    subject_id: str,
    started: tuple[float, float],
) -> dict:
    state = subject.initial_state(subject_id)
    initial_digest = state_digest(state)
    step_index = 0
    raw = bytearray()
    steps = []
    termination = "engineering_failure"
    eos_at_final_check = False
    with torch.no_grad():
        for patch in subject.encoder.patches(prompt.encode()):
            output = subject.step(
                state,
                event=patch,
                goal=goal.encode(),
                records=records,
                growth_gates=provider(step_index),
            )
            state = output.state
            step_index += 1
        previous = EOS
        for _call in range(MAX_CALLS):
            if (
                time.process_time() - started[0] > BUDGET_CPU_SECONDS
                or time.perf_counter() - started[1] > BUDGET_WALL_SECONDS
            ):
                termination = "cpu_wall_budget"
                break
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
                growth_gates=provider(step_index),
            )
            step_index += 1
            probabilities = output.probabilities.detach().cpu().numpy()
            byte = int(torch.argmax(output.probabilities))
            steps.append(
                {
                    "top_byte": byte,
                    "eos_probability": float(probabilities[EOS]),
                    "probabilities": np.round(probabilities, PROBABILITY_DECIMALS).tolist(),
                }
            )
            if len(raw) == MAX_NEW:
                termination = "eos" if byte == EOS else "byte_limit"
                eos_at_final_check = byte == EOS
                break
            if byte == EOS:
                termination = "eos"
                break
            raw.append(byte)
            state = subject.consume_output(output, byte)
            previous = byte
        else:
            termination = "byte_limit"
    return {
        "initial_state_digest": initial_digest,
        "final_state_digest": state_digest(state),
        "prediction_calls": len(steps),
        "raw_bytes": bytes(raw),
        "termination": termination,
        "eos_at_final_check": eos_at_final_check,
        "steps": steps,
    }


def joint_status(first: int, second: int, pass_min: int, absent_max: int) -> str:
    if first >= pass_min and second >= pass_min:
        return "supported"
    if first <= absent_max and second <= absent_max:
        return "absent"
    return "partial"


def scoring_checks() -> list[dict]:
    checks = []
    fixture = targets()
    real = real_pairs()
    permuted = mapping_of(permuted_pairs())
    goals = [describe_goal(entity, place) for entity, _text, place, _place_text in fixture]
    real_actions = [action_sentence(entity, place) for entity, _text, place, _place_text in fixture]
    permuted_actions = [
        action_sentence(permuted[text], permuted[place_text])
        for _entity, text, _place, place_text in fixture
    ]
    expected_len = {len(action.encode()) for action in real_actions}
    goal_len = {len(goal.encode()) for goal in goals}
    checks.append(
        {
            "id": "canonical_action_is_27_bytes",
            "passed": expected_len == {27},
            "observed": sorted(expected_len),
        }
    )
    checks.append(
        {
            "id": "goals_fit_event_budget",
            "passed": max(goal_len) <= 512,
            "observed": sorted(goal_len),
        }
    )
    checks.append(
        {
            "id": "six_actions_distinct",
            "passed": len(set(real_actions)) == 6,
            "observed": len(set(real_actions)),
        }
    )
    checks.append(
        {
            "id": "permuted_actions_differ_from_real",
            "passed": len(set(permuted_actions)) == 6
            and all(
                left != right for left, right in zip(real_actions, permuted_actions, strict=True)
            ),
            "observed": len(set(permuted_actions)),
        }
    )

    oracle_hits = 0
    for entity, _text, place, _place_text in fixture:
        expected = action_sentence(entity, place)
        raw = expected.encode()
        scored = score_action(raw, "eos", expected)
        oracle_hits += int(
            scored["action_exact"]
            and scored["completed_action_exact"]
            and len(raw) <= MAX_NEW
            and len(raw) + 1 <= MAX_CALLS
        )
    checks.append(
        {"id": "oracle_6_of_6_with_eos", "passed": oracle_hits == 6, "observed": oracle_hits}
    )

    wrong_objects = [entity for entity, _ in OBJECT_DESCRIPTIONS.items()]
    wrong_places = [place for place, _ in PLACE_DESCRIPTIONS.items()]

    def other_object(entity, place):
        return action_sentence(wrong_objects[(wrong_objects.index(entity) + 1) % 2], place)

    def other_place(entity, place):
        return action_sentence(entity, wrong_places[(wrong_places.index(place) + 1) % 3])

    families = {
        "wrong_object_0_of_6": other_object,
        "wrong_place_0_of_6": other_place,
        "reversed_0_of_6": lambda entity, place: f"把{place}放到{entity}",
        "multiple_actions_0_of_6": lambda entity, place: (
            action_sentence(entity, place) + action_sentence(entity, place)
        ),
        "table_listing_0_of_6": lambda _entity, _place: "".join(
            f"{text}是{value}。" for text, value in real
        ),
    }
    for check_id, builder in families.items():
        hits = 0
        for entity, _text, place, _place_text in fixture:
            expected = action_sentence(entity, place)
            scored = score_action(builder(entity, place).encode(), "eos", expected)
            hits += int(scored["action_exact"] or scored["completed_action_exact"])
        checks.append({"id": check_id, "passed": hits == 0, "observed": hits})

    fixed = action_sentence(fixture[0][0], fixture[0][2])
    fixed_hits = sum(
        1
        for entity, _text, place, _place_text in fixture
        if score_action(fixed.encode(), "eos", action_sentence(entity, place))["action_exact"]
    )
    checks.append(
        {
            "id": "fixed_true_action_at_most_1_of_6",
            "passed": fixed_hits <= 1,
            "observed": fixed_hits,
        }
    )

    truncation_hits = 0
    prefix_hits = 0
    for entity, _text, place, _place_text in fixture:
        expected = action_sentence(entity, place)
        raw = expected.encode()
        truncation_hits += int(score_action(raw[:16], "byte_limit", expected)["action_exact"])
        prefix_hits += int(score_action(raw[:15], "byte_limit", expected)["action_exact"])
    checks.append(
        {
            "id": "truncation_16_bytes_0_of_6",
            "passed": truncation_hits == 0,
            "observed": truncation_hits,
        }
    )
    checks.append(
        {
            "id": "truncation_valid_utf8_prefix_0_of_6",
            "passed": prefix_hits == 0,
            "observed": prefix_hits,
        }
    )

    padded = score_action(f"  {fixed}。  ".encode(), "eos", fixed)
    checks.append(
        {
            "id": "trailing_punctuation_and_space_stripped",
            "passed": padded["action_exact"],
            "observed": padded["action_exact"],
        }
    )
    prefixed = score_action(f"请{fixed}".encode(), "eos", fixed)
    checks.append(
        {
            "id": "prefix_is_not_stripped",
            "passed": not prefixed["action_exact"],
            "observed": prefixed["action_exact"],
        }
    )
    return checks


def read_row(read_rows: list[dict], condition: str, descriptor: str) -> dict:
    return next(
        row
        for row in read_rows
        if row["condition"] == condition and row["descriptor"] == descriptor
    )


def selected_source(row: dict) -> str:
    if row["top1_ambiguous"]:
        return "ambiguous"
    return row["top1_source_id"] if row["top1_source_id"] is not None else "null"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.21.1 preflight card is write-once")

    torch.set_num_threads(1)
    started = (time.process_time(), time.perf_counter())
    report = {
        "schema": "taiji-composition-preflight-10-21-1-v1",
        "step": "10.21.1",
        "revision": 1,
        "date": "2026-11-01",
        "status": "engineering_failure",
        "declared": {
            "protocol": "64 output bytes, at most 65 predictions; the last call only checks EOS",
            "generation_runs": 18,
            "read_runs": 15,
            "generation_groups": "6 targets x {real records, no records, domain-permuted records}",
            "read_groups": "5 descriptors x {real mapping, domain-permuted mapping, reversed records}",
            "action_exact": (
                "strict equality after stripping leading/trailing whitespace and terminal punctuation"
            ),
            "completed_action_exact": "action_exact and EOS termination",
            "legacy_ordered_hit": "object before place anywhere in the display text; not a gate",
            "generation_gate": (
                "real and permuted completed_action_exact both >=5 -> supported; both <=2 -> absent"
            ),
            "read_gate": (
                "real and permuted target top-1 (null included, ties<=1e-12 wrong) both >=4 -> "
                "supported; both <=1 -> absent"
            ),
            "tie_tolerance": TIE,
            "probability_precision": f"{PROBABILITY_DECIMALS} decimals",
            "training": "none",
            "optimizer_steps": 0,
        },
        "model_readings": {
            "generation": {"status": "not_measured", "face": "legacy_generation_face"},
            "read": {"status": "not_measured", "face": "descriptor_read_diagnostic_upper_bound"},
            "claimable": False,
        },
        "identity": {},
        "instrument_checks": [],
        "generation": {"runs": 0, "rows": [], "summary": {}},
        "read": {"reads": 0, "rows": [], "summary": {}},
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "limits": (
            "Read-only preflight on one synthetic fixture and the 09.8 candidate. The descriptor "
            "read is a diagnostic upper bound with hand-given descriptors, not goal parsing; the "
            "generation task is one fixed template, not free Chinese. No capability, no P2/L3, "
            "no training admission, and no new confirmation data."
        ),
    }
    over_budget = False
    model_started = False
    try:
        bundle_path = ROOT / BUNDLE
        closeout = json.loads((ROOT / CLOSEOUT).read_text(encoding="utf-8"))
        closeout_sha = sha256_text(ROOT / CLOSEOUT)
        manifest = build_manifest(ROOT, bundle_path)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            _manifest, payload = load_candidate(ROOT, manifest_path, bundle_path, "frozen_eval")
        model, state, _optimizer = resume_subject(payload)
        provider = gates_provider(payload["growth_gates"])

        current_hashes = {
            relative: sha256_text(ROOT / relative)
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/constrained_learning.py",
                "taiji/state_migration.py",
                "taiji/native_development.py",
                "taiji/subject_action_scorer.py",
                "taiji/candidate_manifest.py",
                "taiji/slot_composition_head.py",
                "taiji/native_goal_attributes.py",
                "taiji/native_goal_glossary.py",
                "scripts/training/verify_taiji_10_21_1.py",
            )
        }
        closeout_hashes = closeout.get("source_hashes", {})
        source_comparison = {
            relative: {
                "closeout": closeout_hashes.get(relative),
                "current": value,
                "same": closeout_hashes.get(relative) == value,
            }
            for relative, value in current_hashes.items()
            if relative in closeout_hashes
        }
        model_digest = parameter_digest(
            {name: parameter.detach() for name, parameter in model.named_parameters()}
        )
        declared_config = closeout["identity"]["config"]
        effective_config = {key: getattr(model.config, key) for key in declared_config}
        config_extra_keys = sorted(set(vars(model.config)) - set(declared_config))
        identity_checks = [
            {
                "id": "candidate_id",
                "expected": EXPECTED_CANDIDATE,
                "observed": manifest["candidate_id"],
                "passed": manifest["candidate_id"] == EXPECTED_CANDIDATE,
            },
            {
                "id": "bundle_sha256",
                "expected": EXPECTED_BUNDLE_SHA,
                "observed": manifest["provenance"]["sha256"],
                "passed": manifest["provenance"]["sha256"] == EXPECTED_BUNDLE_SHA,
            },
            {
                "id": "bundle_bytes",
                "expected": EXPECTED_BUNDLE_BYTES,
                "observed": manifest["provenance"]["bytes"],
                "passed": manifest["provenance"]["bytes"] == EXPECTED_BUNDLE_BYTES,
            },
            {
                "id": "config_matches_closeout",
                "expected": declared_config,
                "observed": effective_config,
                "extra_keys": config_extra_keys,
                "passed": effective_config == declared_config,
            },
            {
                "id": "subject_id",
                "expected": "taiji-core-09",
                "observed": payload["subject_id"],
                "passed": payload["subject_id"] == "taiji-core-09",
            },
            {
                "id": "growth_parent_units",
                "expected": 256,
                "observed": payload["growth_parent_units"],
                "passed": payload["growth_parent_units"] == 256,
            },
            {
                "id": "growth_unit_branches",
                "expected": [["consensus"] * 3] * 4,
                "observed": payload["growth_unit_branches"],
                "passed": payload["growth_unit_branches"] == [["consensus"] * 3] * 4,
            },
            {
                "id": "growth_gates_table",
                "expected": [{"consensus": 1.0}],
                "observed": payload["growth_gates"],
                "passed": payload["growth_gates"] == [{"consensus": 1.0}],
            },
            {
                "id": "model_parameters_match_candidate",
                "expected": manifest["candidate_id"],
                "observed": model_digest,
                "passed": model_digest == manifest["candidate_id"],
            },
        ]
        report["identity"] = {
            "candidate_id": manifest["candidate_id"],
            "bundle": {
                "path": BUNDLE,
                "sha256": manifest["provenance"]["sha256"],
                "bytes": manifest["provenance"]["bytes"],
            },
            "subject_id": payload["subject_id"],
            "revision": payload["revision"],
            "step": payload["step"],
            "config": effective_config,
            "config_bundle_dict": manifest["config"],
            "config_keys_not_in_closeout": config_extra_keys,
            "closeout_card": {"path": CLOSEOUT, "sha256": closeout_sha},
            "growth_parent_units": payload["growth_parent_units"],
            "growth_unit_branches": payload["growth_unit_branches"],
            "growth_gates": payload["growth_gates"],
            "read_state_origin": (
                "restored bundle state (same H used by 10.21a); generation uses isolated zero-H/F copies"
            ),
            "checks": identity_checks,
            "closeout_source_comparison": source_comparison,
            "source_hashes": current_hashes,
        }

        pairs = real_pairs()
        real_records = records_from_pairs(pairs)
        if tuple(record.raw for record in real_records) != tuple(
            record.raw for record in glossary_records()
        ):
            raise ValueError("local real table differs from the declared glossary module")
        permuted_records = records_from_pairs(permuted_pairs())
        reversed_records = reverse_records(real_records)
        if {(record.source_id, record.raw) for record in reversed_records} != {
            (record.source_id, record.raw) for record in real_records
        }:
            raise ValueError("reversed table must preserve record identities and content")
        real_mapping = mapping_of(pairs)
        permuted_mapping = mapping_of(permuted_pairs())
        tables = {
            "real": (real_records, real_mapping),
            "none": ((), real_mapping),
            "permuted": (permuted_records, permuted_mapping),
        }
        read_tables = {
            "real": real_records,
            "permuted": permuted_records,
            "reversed": reversed_records,
        }

        checks = scoring_checks()
        report["instrument_checks"] = checks
        self_check_passed = all(check["passed"] for check in checks)
        identity_passed = all(check["passed"] for check in identity_checks)
        if not self_check_passed or not identity_passed:
            report["status"] = "instrument_invalid"
        else:
            report["status"] = "preflight_measured"
            rows = []
            model_started = True
            for condition in CONDITIONS:
                records, mapped = tables[condition]
                for entity, text, place, place_text in targets():
                    goal = describe_goal(entity, place)
                    prompt = f"目标是{goal}我应该"
                    expected_action = action_sentence(mapped[text], mapped[place_text])
                    subject_id = f"10.21.1-{condition}-{entity}-{place}"
                    decoded = generation_decode(
                        model, provider, prompt, goal, records, subject_id, started
                    )
                    raw = decoded["raw_bytes"]
                    scored = score_action(raw, decoded["termination"], expected_action)
                    if decoded["termination"] == "cpu_wall_budget":
                        over_budget = True
                    rows.append(
                        {
                            "copy_id": subject_id,
                            "condition": condition,
                            "target": {
                                "object_descriptor": text,
                                "place_descriptor": place_text,
                                "object": entity,
                                "place": place,
                            },
                            "prompt": prompt,
                            "goal": goal,
                            "expected_object": mapped[text],
                            "expected_place": mapped[place_text],
                            "expected_action": expected_action,
                            "records_source": condition,
                            "records_sha256": records_sha(records),
                            "initial_state": "isolated zero h/fast; empty keys/coverage/prior",
                            "initial_state_digest": decoded["initial_state_digest"],
                            "final_state_digest": decoded["final_state_digest"],
                            "prediction_calls": decoded["prediction_calls"],
                            "raw_bytes": len(raw),
                            "raw_hex": raw.hex(),
                            "termination": decoded["termination"],
                            "eos_at_final_check": decoded["eos_at_final_check"],
                            "utf8_valid": scored["utf8_valid"],
                            "display_only": scored["display_only"],
                            "text": scored["text"],
                            "action_exact": scored["action_exact"],
                            "completed_action_exact": scored["completed_action_exact"],
                            "legacy_ordered_hit": legacy_hit(
                                scored["display_only"], mapped[text], mapped[place_text]
                            ),
                            "steps": decoded["steps"],
                        }
                    )
            counts = {}
            for condition in CONDITIONS:
                selected = [row for row in rows if row["condition"] == condition]
                counts[condition] = {
                    "denominator": len(selected),
                    "completed_action_exact": sum(
                        int(row["completed_action_exact"]) for row in selected
                    ),
                    "action_exact": sum(int(row["action_exact"]) for row in selected),
                    "legacy_ordered_hit": sum(int(row["legacy_ordered_hit"]) for row in selected),
                    "eos_terminations": sum(int(row["termination"] == "eos") for row in selected),
                }
            report["generation"] = {
                "runs": len(rows),
                "targets": 6,
                "conditions": list(CONDITIONS),
                "rows": rows,
                "summary": counts,
            }
            generation_status = joint_status(
                counts["real"]["completed_action_exact"],
                counts["permuted"]["completed_action_exact"],
                PASS_HITS,
                ABSENT_HITS,
            )
            report["model_readings"]["generation"] = {
                "status": generation_status,
                "face": "legacy_generation_face",
                "real": counts["real"],
                "permuted": counts["permuted"],
                "no_records": counts["none"],
            }

            read_rows = []
            for condition in READ_CONDITIONS:
                records = read_tables[condition]
                for descriptor in descriptors():
                    with torch.no_grad():
                        reading = descriptor_read(model, state, descriptor, records)
                    read_rows.append(
                        {
                            "condition": condition,
                            "descriptor": descriptor,
                            "records_sha256": records_sha(records),
                            "record_order": [record.source_id for record in records],
                            **reading,
                        }
                    )
            read_counts = {
                condition: sum(
                    int(row["top1_is_target"]) for row in read_rows if row["condition"] == condition
                )
                for condition in READ_CONDITIONS
            }
            read_status = joint_status(
                read_counts["real"], read_counts["permuted"], READ_PASS, READ_ABSENT
            )
            reversed_consistency = {}
            for condition in ("real", "permuted"):
                forward = {
                    descriptor: selected_source(read_row(read_rows, condition, descriptor))
                    for descriptor in descriptors()
                }
                backward = {
                    descriptor: selected_source(read_row(read_rows, "reversed", descriptor))
                    for descriptor in descriptors()
                }
                reversed_consistency[condition] = {
                    "same_selected_source": sum(
                        int(forward[descriptor] == backward[descriptor])
                        for descriptor in descriptors()
                    ),
                    "denominator": 5,
                    "reversed_target_top1": sum(
                        int(read_row(read_rows, "reversed", descriptor)["top1_is_target"])
                        for descriptor in descriptors()
                    ),
                }
            query_gain = []
            for condition in ("real", "permuted"):
                masses = {
                    descriptor: read_row(read_rows, condition, descriptor)["record_masses"]
                    for descriptor in descriptors()
                }
                for index, descriptor in enumerate(descriptors()):
                    own = masses[descriptor][index]
                    others = [
                        masses[other][index] for other in descriptors() if other != descriptor
                    ]
                    query_gain.append(
                        {
                            "condition": condition,
                            "descriptor": descriptor,
                            "own_record_mass": own,
                            "mean_other_descriptor_mass": sum(others) / len(others),
                            "query_gain": own - sum(others) / len(others),
                        }
                    )
            irrelevant_exchange = []
            for descriptor in descriptors():
                real_row = read_row(read_rows, "real", descriptor)
                permuted_row = read_row(read_rows, "permuted", descriptor)
                irrelevant_exchange.append(
                    {
                        "descriptor": descriptor,
                        "real_target_mass": real_row["target_mass"],
                        "permuted_target_mass": permuted_row["target_mass"],
                        "delta": permuted_row["target_mass"] - real_row["target_mass"],
                        "same_selected_source": selected_source(real_row)
                        == selected_source(permuted_row),
                    }
                )
            report["read"] = {
                "reads": len(read_rows),
                "descriptors": 5,
                "conditions": list(READ_CONDITIONS),
                "rows": read_rows,
                "summary": {
                    "target_top1": read_counts,
                    "denominator": 5,
                    "reversed_consistency": reversed_consistency,
                    "query_gain": query_gain,
                    "irrelevant_exchange": irrelevant_exchange,
                    "real_mapping": real_mapping,
                    "permuted_mapping": permuted_mapping,
                    "source_identity": {
                        condition: {
                            record.source_id: record.raw.decode("utf-8")
                            for record in read_tables[condition]
                        }
                        for condition in READ_CONDITIONS
                    },
                },
            }
            report["model_readings"]["read"] = {
                "status": read_status,
                "face": "descriptor_read_diagnostic_upper_bound",
                "real_target_top1": read_counts["real"],
                "permuted_target_top1": read_counts["permuted"],
                "reversed_target_top1": read_counts["reversed"],
            }

            log_problems = []
            for row in rows:
                if row["prediction_calls"] > MAX_CALLS:
                    log_problems.append(f"{row['copy_id']}: {row['prediction_calls']} calls")
                if row["raw_bytes"] > MAX_NEW:
                    log_problems.append(f"{row['copy_id']}: {row['raw_bytes']} bytes")
                if len(row["steps"]) != row["prediction_calls"]:
                    log_problems.append(f"{row['copy_id']}: step log length")
                for step_index, step in enumerate(row["steps"]):
                    probabilities = step["probabilities"]
                    if len(probabilities) != 257:
                        log_problems.append(f"{row['copy_id']}:{step_index} width")
                    total = float(sum(probabilities))
                    if abs(total - 1.0) > 1e-5:
                        log_problems.append(f"{row['copy_id']}:{step_index} sum={total}")
                    if not np.all(np.isfinite(probabilities)):
                        log_problems.append(f"{row['copy_id']}:{step_index} non-finite")
            parameter_after = parameter_digest(
                {name: parameter.detach() for name, parameter in model.named_parameters()}
            )
            report["log_checks"] = {
                "runs": len(rows),
                "reads": len(read_rows),
                "probabilities_width": 257,
                "problems": log_problems,
                "parameters_unchanged": parameter_after == manifest["candidate_id"],
                "optimizer_steps": 0,
            }
            if log_problems or parameter_after != manifest["candidate_id"]:
                report["status"] = "instrument_invalid"
            elif over_budget:
                report["status"] = "budget_stop"
            report["model_readings"]["claimable"] = report["status"] == "preflight_measured"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "engineering_failure"
        report["engineering_failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "model_runs_started": model_started,
        }
    cpu_seconds = time.process_time() - started[0]
    wall_seconds = time.perf_counter() - started[1]
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        over_budget = True
        if report["status"] == "preflight_measured":
            report["status"] = "budget_stop"
    report["cpu_seconds"] = cpu_seconds
    report["wall_seconds"] = wall_seconds
    prior_attempts = []
    for card in sorted(ROOT.glob("reports/taiji_core_10_21_1_preflight_20261101_v*.json")):
        prior_report = json.loads(card.read_text(encoding="utf-8"))
        prior_attempts.append(
            {
                "path": str(card.relative_to(ROOT)),
                "status": prior_report.get("status"),
                "cpu_seconds": prior_report.get("cpu_seconds"),
            }
        )
    report["prior_attempts"] = prior_attempts
    report["cpu_seconds_including_prior_attempts"] = cpu_seconds + sum(
        float(attempt["cpu_seconds"] or 0.0) for attempt in prior_attempts
    )
    report["over_budget"] = over_budget
    report["model_involved"] = bool(report["generation"]["runs"])
    report["parameters_changed"] = False
    report["capability_claimed"] = False
    report["training_admitted"] = False
    report["data_splits_read"] = ["synthetic fixture: native_goal_attributes/native_goal_glossary"]
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
                "status": report["status"],
                "generation": report["model_readings"]["generation"].get("status"),
                "read": report["model_readings"]["read"].get("status"),
                "runs": report["generation"]["runs"],
                "reads": report["read"]["reads"],
                "cpu_seconds": round(cpu_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
