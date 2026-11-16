"""22.5.A.5: measure the 22.5.1 parent protocol on the strongest configuration.

What this runner does, in order, per seed:

1. Build the frozen subject and the frozen calibrated read, and turn the read
   into the ``read-code-v1`` one-hot code exactly as A.2 does.  The byte head is
   trained on the A task (the read names the position, the head writes it), which
   is the configuration that already measured 32/32.
2. Train 600 updates on the train split, evaluating fit+train at three doses.
   The trainable set is A.2's eight tensors **plus the record channel** - the one
   structural increment this card adds, and the only way a released record can
   reach the bytes.
3. Measure the parent protocol on task B (fact revision):
     - first pass without the record (wrong by construction, measured 0/24);
     - the environment legally releases a record through ``allow_feedback`` and
       the ``feedback_record_ids`` gate, one error at a time;
     - the error is re-asked immediately; a single update is taken only if the
       re-ask is still wrong, and a wrong update is rolled back to the previous
       good parameters;
     - a frozen-update control receives the same exposure and makes no update,
       and a second pass over every B example gives the gain;
     - an A.4-style arm (eight accumulated updates per error) runs on the same
       degraded first pass, purely for attribution;
     - a counter-fact arm releases a record that contradicts the released fact,
       so a subject that never reads the record cannot track it.
4. Check retention on the A task (monitor only: the gate probe measured the A
   distribution to be invariant to the copy gate without a record, so this check
   cannot fail by construction and may not be cited as evidence).
5. Save the bundle, reload it in an independent process, and require the 32
   A-task rounds to come back byte-identical.

Modes: ``smoke`` (zero optimizer steps: builds the protocol and checks the gates
on the frozen subject), ``run`` (formal training), ``recover`` (independent
readback).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_events import observation_intent
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_subject import U2Packet, U2Record

CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_a5_protocol_run_20261116.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_a5_protocol_smoke_20261116.json"
LABELS = ("左边", "中间", "右边", "未知")
SUFFIX = "现在在哪里？"
RELEASE_ID = "a5-release"


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("events", ROOT / "taiji/u2_events.py"),
        ("a2_runner", ROOT / "scripts/training/run_a2_byte_20261115.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("columns", ROOT / "taiji/u2_observation_columns.py"),
    ):
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cpu_cap") or not contract.get("wall_cap"):
            raise ValueError("missing_measured_cap")


# --------------------------------------------------------------------------- data


def task_b_examples(data: dict, contract: dict) -> tuple[list[dict], list[str]]:
    """Fit examples whose target is a position label, with the revised label.

    The revised label is the next position in the cycle, and the counter-fact
    label is the third one, so ``revised`` never equals ``counter`` and neither
    equals the frozen truth.
    """
    out = []
    skipped = []
    order = LABELS[:3]
    for example in data["fit"]:
        truth = example["target"].decode()
        if truth not in order:
            skipped.append(example["id"])
            continue
        index = order.index(truth)
        revised = order[(index + 1) % 3]
        counter = order[(index + 2) % 3]
        public = p4_2.column_public(example)
        entity = public["request"].decode()[: -len(SUFFIX)]
        templates = contract["task_b"]["templates"]
        records = []
        for template in templates:
            raw = template.format(entity=entity, label=revised).encode()
            records.append(raw)
        offsets = [raw.index(revised.encode()) for raw in records]
        if offsets != list(contract["task_b"]["template_offsets"]):
            raise ValueError(f"template_offset_mismatch:{example['id']}:{offsets}")
        if len(set(offsets)) != len(templates):
            raise ValueError("template_offsets_not_distinct")
        counter_records = [
            template.format(entity=entity, label=counter).encode() for template in templates
        ]
        out.append(
            {
                "example": example,
                "entity": entity,
                "truth": truth,
                "revised": revised,
                "counter": counter,
                "target": revised.encode(),
                "counter_target": counter.encode(),
                "records": records,
                "counter_records": counter_records,
                "offsets": offsets,
            }
        )
    if len(out) < 4:
        raise ValueError("task_b_denominator_below_four")
    return out, skipped


def record_of(raw: bytes, identity: str) -> U2Record:
    return U2Record(
        record_id=identity,
        raw=raw,
        version=0,
        source="sensor",
        scope="world",
        observed_at=0.0,
        writable=True,
    )


# ------------------------------------------------------------------------- episode


def episode(
    model,
    public: dict,
    identity: str,
    records: list[U2Record] | None,
    release: bool,
    release_ids: tuple[str, ...] | None = None,
):
    """The byte pipeline's own prepare, plus the record channel and its legal release.

    The record has to be in the state before ``begin_utterance``: the utterance
    boundary rebuilds the cursor, and the cursor is what the copy branch reads.
    The release is a second observation packet whose ``feedback_record_ids``
    names the record; the subject refuses an id that has no matching writable
    record ("feedback record not legally released"), which is what makes this an
    environment result rather than teacher forcing.
    """
    state = model.set_goal(model.initial(identity), public["goal"])
    state = observation_intent(model, state, "recorded-observation", 0.05)
    state, _ = model.assimilate(
        state,
        U2Packet(
            "recorded-public-frame",
            "sensor",
            "world",
            public["t_start"],
            public["t_end"],
            0.0,
            public["observations"],
            public["coordinates"],
            public["outcome"],
            "recorded-observation",
        ),
    )
    state = model.set_request(state, "current-request", public["request"])
    if records:
        state = model.set_records(state, records)
    if release:
        state = observation_intent(model, state, RELEASE_ID, 0.05)
        state, _ = model.assimilate(
            state,
            U2Packet(
                f"{RELEASE_ID}-packet",
                "sensor",
                "world",
                public["t_end"],
                public["t_end"] + 0.05,
                0.0,
                public["observations"],
                public["coordinates"],
                public["outcome"],
                RELEASE_ID,
                feedback_record_ids=(
                    release_ids
                    if release_ids is not None
                    else tuple(record.record_id for record in records or ())
                ),
            ),
        )
    return model.begin_utterance(state)


def answer_bytes(model, state, code: torch.Tensor | None) -> dict:
    cap = int(model.c.max_output_bytes) + 1
    content: list[int] = []
    eos = False
    with torch.no_grad():
        context = a2.reading_code(model, code) if code is not None else _nullcontext()
        with context:
            for _ in range(cap):
                probability, _, details = model.output(
                    state, model.language_goal(state), state["prediction_action"]
                )
                symbol = int(probability.argmax())
                if symbol == 256:
                    eos = True
                    break
                content.append(symbol)
                state = model.accept_byte(state, symbol, details)
    raw = bytes(content)
    try:
        raw.decode("utf-8", errors="strict")
        valid = True
    except UnicodeDecodeError:
        valid = False
    return {"raw": raw, "eos": eos, "valid_utf8": valid}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def a_round(model, example: dict, code: torch.Tensor) -> dict:
    public = p4_2.column_public(example)
    state = episode(model, public, example["id"], None, False)
    got = answer_bytes(model, state, code)
    return {
        "id": example["id"],
        "target": example["target"].decode(),
        "output_hex": got["raw"].hex(),
        "eos": got["eos"],
        "valid_utf8": got["valid_utf8"],
        "correct": bool(got["eos"] and got["valid_utf8"] and got["raw"] == example["target"]),
    }


def b_round(model, item: dict, code: torch.Tensor, template: int, counter: bool = False):
    public = p4_2.column_public(item["example"])
    raw = item["counter_records"][template] if counter else item["records"][template]
    label = item["counter_target"] if counter else item["target"]
    records = [record_of(raw, "a5-fact")]
    state = episode(model, public, item["example"]["id"], records, True)
    got = answer_bytes(model, state, code)
    return {
        "id": item["example"]["id"],
        "template": template,
        "target": label.decode(),
        "output_hex": got["raw"].hex(),
        "eos": got["eos"],
        "valid_utf8": got["valid_utf8"],
        "correct": bool(got["eos"] and got["valid_utf8"] and got["raw"] == label),
        "follows_truth": bool(got["raw"] == item["target"]),
    }


def b_pass(model, items: list[dict], codes: dict, template_of) -> list[dict]:
    return [
        b_round(model, item, codes[item["example"]["id"]], template_of(index))
        for index, item in enumerate(items)
    ]


def accuracy(rows: list[dict]) -> float:
    return sum(int(row["correct"]) for row in rows) / len(rows)


def rows_digest(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# ----------------------------------------------------------------------- training


def build_subject(contract: dict) -> torch.nn.Module:
    candidate = contract["candidate"]
    bundle = ROOT / candidate["encoder_bundle"]
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    torch.manual_seed(int(candidate["seed"]))
    model = make_model("native", int(candidate["seed"]))
    model.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != candidate["parameters"]:
        raise ValueError("candidate_parameter_count_mismatch")
    if len(list(model.named_parameters())) != candidate["tensors"]:
        raise ValueError("candidate_tensor_count_mismatch")
    return model


def declared_trainable(contract: dict) -> list[str]:
    return sorted([*contract["emission_parameters"], *contract["record_parameters"]])


def set_trainable(model: torch.nn.Module, contract: dict) -> list[str]:
    declared = set(declared_trainable(contract))
    found = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
        if name in declared:
            found.append(name)
    if sorted(found) != sorted(declared):
        raise ValueError("trainable_set_mismatch")
    tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    scalars = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if tensors != int(contract["trainable_tensor_count"]):
        raise ValueError("trainable_tensor_mismatch")
    if scalars != int(contract["trainable_scalar_count"]):
        raise ValueError("trainable_scalar_mismatch")
    return sorted(found)


def optimizer_for(parameters, spec: dict):
    return torch.optim.Adam(
        parameters,
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )


def snapshot(model: torch.nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in set(names)
    }


def restore(model: torch.nn.Module, saved: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in saved:
                parameter.copy_(saved[name])


def train_step(model, optimizer, items: list[dict], required: set[str]) -> tuple[float, list[str]]:
    """One A-task update; record which declared tensors the A loss never touches.

    The A task has no record in the cursor, so the record channel is outside the
    A graph and its gradient is None.  That is a property of the task, not a
    broken graph, so it is measured and reported instead of being asserted away;
    what must not happen is an *emission* tensor arriving without a gradient.
    """
    optimizer.zero_grad(set_to_none=True)
    loss = a2.emission_loss(model, items, use_code=True)
    loss.backward()
    inert = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            inert.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError(f"nonfinite_a5_gradient:{name}")
    missing = sorted(set(required) & set(inert))
    if missing:
        raise ValueError(f"emission_gradient_absent:{missing}")
    optimizer.step()
    model.parameter_revision += 1
    return float(loss.detach()), inert


def record_loss(model, public: dict, identity: str, raw: bytes, target: bytes, code):
    """Byte loss on the answer, conditioned on the legally released record."""
    records = [record_of(raw, "a5-fact")]
    terms = []
    with a2.reading_code(model, code):
        state = episode(model, public, identity, records, True)
        for symbol in [*target, 256]:
            probability, _, details = model.output(
                state, model.language_goal(state), state["prediction_action"]
            )
            terms.append(-probability[symbol])
            with torch.no_grad():
                state = model.accept_byte(state, symbol, details)
    return torch.stack(terms).mean()


# ----------------------------------------------------------------------- protocol


def protocol_one_at_a_time(model, items, codes, contract, budget, template_of) -> dict:
    """Release one error at a time, re-ask, one update at most, roll back on harm."""
    names = declared_trainable(contract)
    optimizer = optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    released = 0
    updates = 0
    rollbacks = 0
    corrected = 0
    trace = []
    first_errors = [item for item in items]
    for index, item in enumerate(first_errors):
        budget()
        public = p4_2.column_public(item["example"])
        code = codes[item["example"]["id"]]
        template = template_of(index)
        asked = b_round(model, item, code, template)
        released += 1
        if asked["correct"]:
            corrected += 1
            trace.append(
                {"id": item["example"]["id"], "template": template, "path": "released_only"}
            )
            continue
        saved = snapshot(model, names)
        optimizer.zero_grad(set_to_none=True)
        loss = record_loss(
            model,
            public,
            item["example"]["id"],
            item["records"][template],
            item["target"],
            code,
        )
        loss.backward()
        optimizer.step()
        model.parameter_revision += 1
        updates += 1
        again = b_round(model, item, code, template)
        if again["correct"]:
            corrected += 1
            trace.append({"id": item["example"]["id"], "template": template, "path": "updated"})
        else:
            restore(model, saved)
            model.parameter_revision += 1
            rollbacks += 1
            # The Adam moments belong to the rejected step.  Keeping them would
            # mean the rollback only restored the weights, not the trajectory.
            optimizer = optimizer_for(
                [p for p in model.parameters() if p.requires_grad],
                contract["feedback_optimizer"],
            )
            trace.append({"id": item["example"]["id"], "template": template, "path": "rolled_back"})
    return {
        "releases": released,
        "updates": updates,
        "rollbacks": rollbacks,
        "corrected": corrected,
        "correction_rate": corrected / len(first_errors),
        "trace": trace,
    }


def protocol_accumulated_a4(model, items, codes, contract, budget, template_of) -> dict:
    """A.4's protocol on the same first pass: eight accumulated updates per error."""
    optimizer = optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    updates = 0
    for index, item in enumerate(items):
        public = p4_2.column_public(item["example"])
        code = codes[item["example"]["id"]]
        template = template_of(index)
        for _ in range(int(contract["feedback_updates_a4"])):
            budget()
            optimizer.zero_grad(set_to_none=True)
            loss = record_loss(
                model,
                public,
                item["example"]["id"],
                item["records"][template],
                item["target"],
                code,
            )
            loss.backward()
            optimizer.step()
            model.parameter_revision += 1
            updates += 1
    return {"updates": updates}


def bootstrap_lower_bound(pairs: list[tuple[int, int]], contract: dict) -> dict:
    """World-paired bootstrap: the unit is the example, not the row."""
    count = int(contract["plan"]["bootstrap_resamples"])
    generator = random.Random(int(contract["plan"]["bootstrap_seed"]))
    differences = [learned - control for learned, control in pairs]
    means = []
    for _ in range(count):
        sample = [differences[generator.randrange(len(differences))] for _ in differences]
        means.append(sum(sample) / len(sample))
    means.sort()
    index = max(0, int(0.025 * count) - 1)
    return {
        "unit": "example",
        "pairs": len(pairs),
        "observed_mean": round(statistics.fmean(differences), 10),
        "resamples": count,
        "lower_bound_95": round(means[index], 10),
        "upper_bound_95": round(means[-index - 1], 10),
    }


def decide(summary: dict) -> str:
    gates = summary["gates"]
    if gates["main"] == "failed":
        # The card enumerates no exit for a main-gate failure, so this one is
        # named rather than folded into parent_protocol_partial, which the card
        # defines as "main satisfied, something else not".
        return "main_gate_unmet"
    if gates["correction"] == "not_identifiable_by_design":
        return "not_identifiable_by_design"
    if gates["correction"] == "failed" or gates["gain"] == "failed":
        # Three different failures look alike in the gate table and must not be
        # merged: the A.4-style arm is the control for what is reachable at all on
        # the same first pass.  If it also reaches zero, the limit is capability;
        # if it climbs while the new protocol stays at zero, the new protocol is
        # what suppressed the learning.
        a4 = summary["a4_second_pass"]
        learned = summary["learned_second_pass"]
        if a4 <= 0.0:
            return "protocol_defect_is_capability_gap"
        if a4 > learned + 0.05:
            return "correction_protocol_suppressed_learning"
        return "feedback_protocol_fixed_but_gain_absent"
    if (
        gates["gain"] == "passed"
        and gates["lower_bound"] == "passed"
        and gates["three_seed"] == "passed"
        and gates["retention"] == "passed"
        and gates["recovery"] == "passed"
    ):
        return "parent_protocol_measured"
    return "parent_protocol_partial"


# ---------------------------------------------------------------------------- run


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    root = ROOT / contract["run_root"]
    root.mkdir(parents=True, exist_ok=True)

    def budget() -> None:
        if time.process_time() - cpu_start > float(contract["cpu_cap"]):
            raise BudgetStop("a5_budget_stop")
        if time.perf_counter() - wall_start > float(contract["wall_cap"]):
            raise BudgetStop("a5_wall_stop")

    items, skipped = task_b_examples(data, contract)
    template_of = lambda index: index % len(contract["task_b"]["templates"])  # noqa: E731

    base_model = build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    if not torch.allclose(prior, torch.tensor(contract["candidate"]["prior"]), atol=1e-6, rtol=0):
        raise ValueError("prior_mismatch")
    codes = a2.code_by_id(base_model, head, prior, data["train"] + data["fit"], contract)
    results = []
    resumed = []
    phase: dict[str, float] = {}

    def mark(name: str, start: float) -> None:
        phase[name] = round(phase.get(name, 0.0) + (time.process_time() - start), 6)

    for seed in [int(value) for value in contract["plan"]["seeds"]]:
        budget()
        seed_root = root / f"seed-{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        cached = seed_root / "result.json"
        if cached.exists():
            results.append(json.loads(cached.read_text(encoding="utf-8")))
            resumed.append(seed)
            continue
        # A seed that was interrupted has no result.json, so its bundles are not
        # a reading and must not block the re-run: checkpoints are create-once,
        # which turns a stale bundle into an error instead of an overwrite.
        for stale in ("bundle-pre-feedback.pt", "bundle-post-feedback.pt"):
            leftover = seed_root / stale
            if leftover.exists():
                leftover.unlink()
        seed_start = time.process_time()
        # Only the primary seed carries the learning curve.  The intermediate
        # doses are a diagnostic block that no gate reads, and on a half-trained
        # head they run to the 65-byte cap and cost more than everything else in
        # the seed together.  Seeds 2 and 3 measure the final dose only; that is
        # declared in the contract rather than discovered in a budget stop.
        doses = (
            [int(value) for value in contract["dose_points"]]
            if seed in [int(value) for value in contract["execution"]["intermediate_doses_seeds"]]
            else [max(int(value) for value in contract["dose_points"])]
        )
        model = copy.deepcopy(base_model)
        names = set_trainable(model, contract)
        positions = a2.cache_positions(model, data["train"], codes)
        # The subject, the read and the code are frozen, so the only thing a run
        # seed can vary is the order in which the cached positions are consumed.
        # That is a weaker replication than three initialisations and the report
        # says so; treating identical runs as three seeds would be decorative.
        order = list(range(len(positions)))
        random.Random(seed).shuffle(order)
        optimizer = optimizer_for(
            [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
        )
        losses = []
        inert_first_step: list[str] = []
        train_start = time.process_time()
        dose_cpu = 0.0
        cursor = 0
        dose_rows = []
        for update in range(1, int(contract["steps"]) + 1):
            budget()
            batch = [
                positions[order[(cursor + offset) % len(order)]]
                for offset in range(int(contract["batch"]))
            ]
            cursor = (cursor + int(contract["batch"])) % len(order)
            step_loss, inert = train_step(
                model, optimizer, batch, set(contract["emission_parameters"])
            )
            losses.append(step_loss)
            if update == 1:
                inert_first_step = inert
            if update not in doses:
                continue
            budget()
            dose_start = time.process_time()
            budget_rounds = int(contract["execution"]["eval_rounds_by_dose"][str(update)])
            fit_rows = [a_round(model, example, codes[example["id"]]) for example in data["fit"]]
            # An intermediate dose only needs the fit split: the extra 64 train
            # rounds tripled the price of a partly trained head (it runs to the
            # 65-byte cap) without feeding any gate.  The final dose still reads
            # all 96 rounds.
            train_rows = (
                [a_round(model, example, codes[example["id"]]) for example in data["train"]]
                if budget_rounds > len(data["fit"])
                else []
            )
            dose_rows.append(
                {
                    "update": update,
                    "fit_correct": sum(int(row["correct"]) for row in fit_rows),
                    "fit_accuracy": accuracy(fit_rows),
                    "train_correct": (
                        sum(int(row["correct"]) for row in train_rows) if train_rows else None
                    ),
                    "train_accuracy": accuracy(train_rows) if train_rows else None,
                    "fit_rounds": len(fit_rows) + len(train_rows),
                }
            )
            dose_cpu += time.process_time() - dose_start
        mark("training", train_start)
        phase["dose_eval"] = round(phase.get("dose_eval", 0.0) + dose_cpu, 6)
        final = dose_rows[-1]
        post_start = time.process_time()
        # The bundle is taken at the end of A training, so the published main
        # gate reproduces from it; the post-feedback bundle is saved separately.
        pre_feedback = seed_root / "bundle-pre-feedback.pt"
        pre_digest = p4_2.save_checkpoint(
            pre_feedback,
            {
                "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
                "update": int(contract["steps"]),
                "revision": int(model.parameter_revision),
                "data_digest": contract["data_digest"],
            },
        )

        # ---- task B first pass, without the record (wrong by construction)
        budget()
        first = b_pass(model, items, codes, template_of)
        errors = [items[index] for index, row in enumerate(first) if not row["correct"]]
        error_templates = [
            template_of(index) for index, row in enumerate(first) if not row["correct"]
        ]

        # ---- control: same exposure, zero updates, zero state writes
        control = copy.deepcopy(model)
        control_before = value_digest(control.state_dict())
        budget()
        control_rows = b_pass(control, items, codes, template_of)
        control_after = value_digest(control.state_dict())

        # ---- learned arm: one release, one re-ask, at most one update
        learned = copy.deepcopy(model)
        protocol = (
            protocol_one_at_a_time(
                learned,
                errors,
                codes,
                contract,
                budget,
                lambda index, table=error_templates: table[index],
            )
            if errors
            else {
                "releases": 0,
                "updates": 0,
                "rollbacks": 0,
                "corrected": 0,
                "correction_rate": 0.0,
                "trace": [],
            }
        )
        budget()
        learned_rows = b_pass(learned, items, codes, template_of)

        # ---- counter-fact arm: the record says the opposite of what was released
        budget()
        counter_rows = [
            b_round(learned, item, codes[item["example"]["id"]], template_of(index), counter=True)
            for index, item in enumerate(items)
        ]
        follows_counter = sum(
            1
            for row, item in zip(counter_rows, items, strict=True)
            if row["output_hex"] == item["counter_target"].hex()
        )

        # ---- A.4-style attribution arm on the same first pass
        #
        # Attribution, not a gate: it is measured on the primary seeds only.
        # Measured on seed 20261111, this arm alone was 73 percent of a seed's
        # cost (192 record-conditioned updates, each rebuilding an episode with
        # a legal release), and the gate items do not read it.
        budget()
        a4_arm = copy.deepcopy(model)
        measure_a4 = seed in [int(value) for value in contract["execution"]["a4_arm_seeds"]]
        a4_stats = (
            protocol_accumulated_a4(
                a4_arm,
                errors,
                codes,
                contract,
                budget,
                lambda index, table=error_templates: table[index],
            )
            if errors and measure_a4
            else {"updates": 0}
        )
        budget()
        a4_rows = b_pass(a4_arm, items, codes, template_of) if measure_a4 else []

        # ---- retention monitor on the A task (cannot fail by construction)
        budget()
        retained_rows = [a_round(learned, example, codes[example["id"]]) for example in data["fit"]]
        damage = accuracy(
            [a_round(model, example, codes[example["id"]]) for example in data["fit"]]
        ) - accuracy(retained_rows)

        post_feedback = seed_root / "bundle-post-feedback.pt"
        post_digest = p4_2.save_checkpoint(
            post_feedback,
            {
                "model": {
                    key: value.detach().clone() for key, value in learned.state_dict().items()
                },
                "update": "feedback",
                "revision": int(learned.parameter_revision),
            },
        )

        pairs = [
            (int(learned_row["correct"]), int(control_row["correct"]))
            for learned_row, control_row in zip(learned_rows, control_rows, strict=True)
        ]
        results.append(
            {
                "seed": seed,
                "doses": dose_rows,
                "final": final,
                "first_pass": {
                    "correct": sum(int(row["correct"]) for row in first),
                    "denominator": len(first),
                    "accuracy": accuracy(first),
                    "error_ids": [item["example"]["id"] for item in errors],
                },
                "feedback": {
                    **protocol,
                    "control_second_pass": accuracy(control_rows),
                    "learned_second_pass": accuracy(learned_rows),
                    "control_updates": 0,
                    "control_state_unchanged": control_before == control_after,
                    "gain": accuracy(learned_rows) - accuracy(control_rows),
                    "a4_updates": int(a4_stats["updates"]),
                    "a4_second_pass": accuracy(a4_rows) if a4_rows else None,
                },
                "counter_fact": {
                    "follows_counter": follows_counter,
                    "denominator": len(items),
                    "tracks_the_record": follows_counter / len(items),
                },
                "retention": {
                    "a_fit_before": final["fit_accuracy"],
                    "a_fit_after": accuracy(retained_rows),
                    "damage": damage,
                },
                "pairs": pairs,
                "a_fit_after_feedback_digest": rows_digest(retained_rows),
                "bundle_pre_feedback": str(pre_feedback.relative_to(ROOT)).replace("\\", "/"),
                "bundle_pre_feedback_sha256": pre_digest,
                "bundle_post_feedback": str(post_feedback.relative_to(ROOT)).replace("\\", "/"),
                "bundle_post_feedback_sha256": post_digest,
                "loss_first10": statistics.fmean(losses[:10]),
                "loss_last10": statistics.fmean(losses[-10:]),
                "trainable_names": names,
                "record_parameters_inert_in_a_training": inert_first_step,
                "doses_measured": doses,
                "seed_cpu": round(time.process_time() - seed_start, 6),
            }
        )
        mark("protocol", post_start)
        (seed_root / "result.json").write_text(
            json.dumps(results[-1], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    primary = results[0]
    pooled = [pair for row in results for pair in row["pairs"]]
    lower = bootstrap_lower_bound(pooled, contract)
    per_seed_lower = [bootstrap_lower_bound(row["pairs"], contract) for row in results]
    gains = [row["feedback"]["gain"] for row in results]
    criteria = contract["criteria"]
    main_gate = criteria["main_gate"]
    gates = {
        "main": "passed" if primary["final"]["fit_accuracy"] >= main_gate else "failed",
        "correction": (
            "not_identifiable_by_design"
            if primary["first_pass"]["denominator"] - primary["first_pass"]["correct"] < 4
            else (
                "passed"
                if primary["feedback"]["correction_rate"] >= criteria["correction_gate"]
                else "failed"
            )
        ),
        "gain": (
            "passed" if primary["feedback"]["gain"] >= criteria["feedback_gain_gate"] else "failed"
        ),
        "lower_bound": "passed" if lower["lower_bound_95"] > 0 else "failed",
        "three_seed": (
            "passed"
            if all(gain > 0 for gain in gains) or all(gain < 0 for gain in gains)
            else "failed"
        ),
        "retention": "passed" if primary["retention"]["damage"] <= 0.02 else "failed",
    }
    # The recovery check must compare a fresh process against the rows this
    # process actually produced.  Recomputing from the bundle first and then
    # comparing to the bundle would be a tautology, so the in-process digest of
    # the post-feedback A rounds is what the independent process is held to.
    independent = recover_rounds(ROOT / primary["bundle_post_feedback"], contract)
    independent["in_process_digest"] = primary["a_fit_after_feedback_digest"]
    independent["rows_equal"] = independent["digest"] == primary["a_fit_after_feedback_digest"]
    gates["recovery"] = "passed" if independent["rows_equal"] else "failed"

    summary = {
        "gates": gates,
        "a4_second_pass": primary["feedback"]["a4_second_pass"],
        "learned_second_pass": primary["feedback"]["learned_second_pass"],
        "gains": gains,
    }
    verdict = decide(summary)
    payload = {
        "schema": "taiji-u2-a5-protocol-run-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": verdict,
        "seeds": results,
        "denominator": {
            "task_b_examples": len(items),
            "task_b_skipped_unknown": len(skipped),
            "task_b_skipped_ids": skipped,
            "fit_rounds": len(data["fit"]),
            "train_rounds": len(data["train"]),
        },
        "templates": contract["task_b"]["templates"],
        "template_offsets": contract["task_b"]["template_offsets"],
        "bootstrap": {"pooled": lower, "per_seed": per_seed_lower},
        "gates": gates,
        "gain_by_seed": gains,
        "recovery": independent,
        "resumed_seeds": resumed,
        "phase_cpu": phase,
        "criteria": criteria,
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "unit_cpu_measured": {
                "training_per_seed": phase.get("training"),
                "dose_eval_all_seeds": phase.get("dose_eval"),
                "protocol_per_seed_block": phase.get("protocol"),
            },
            "wall": round(time.perf_counter() - wall_start, 6),
            "cap_cpu": contract["cpu_cap"],
            "cap_wall": contract["wall_cap"],
        },
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "scope": (
            "the 22.5.1 parent protocol measured on the byte world; passing does not "
            "sign an A parent (22.5.0 is still open) and the retention check may not be "
            "cited as evidence"
        ),
    }
    RUN_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "gains": gains,
                "lower_bound": lower["lower_bound_95"],
                "cost": payload["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def recover_rounds(path: Path, contract: dict) -> dict:
    """Re-ask the 32 A rounds in a fresh process and digest what came back."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__)), "--recover", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"recovery_failed:{completed.stderr[-400:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def recover(path: Path, contract: dict) -> str:
    torch.set_num_threads(1)
    data = atomic_load(ROOT / contract["data_root"])
    model = build_subject(contract)
    model.load_state_dict(torch.load(path, weights_only=False)["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["fit"], contract)
    rows = [a_round(model, example, codes[example["id"]]) for example in data["fit"]]
    return json.dumps({"rows": rows, "digest": rows_digest(rows)}, ensure_ascii=False)


def smoke(contract: dict) -> int:
    """Build the whole protocol on the frozen subject: zero optimizer steps."""
    guards(contract, "smoke")
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    data = atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    items, skipped = task_b_examples(data, contract)
    model = build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)
    names = set_trainable(model, contract)

    template_of = lambda index: index % len(contract["task_b"]["templates"])  # noqa: E731
    a_rows = [a_round(model, example, codes[example["id"]]) for example in data["fit"]]
    first = b_pass(model, items, codes, template_of)
    # The legal release must be refused for a record that was never in the state.
    public = p4_2.column_public(items[0]["example"])
    record = record_of(items[0]["records"][0], "a5-fact")
    legal_state = episode(model, public, items[0]["example"]["id"], [record], True)
    legal = {
        "released": list(legal_state["released_feedback_records"]),
        "records": len(legal_state["records"]),
        "fast_norm": round(float(legal_state["F"].detach().norm()), 10),
    }
    refusal = ""
    try:
        episode(
            model,
            public,
            items[0]["example"]["id"],
            [record],
            True,
            release_ids=("a5-missing",),
        )
        refusal = "accepted"
    except ValueError as error:
        refusal = str(error)
    counter_rows = [
        b_round(model, item, codes[item["example"]["id"]], template_of(index), counter=True)
        for index, item in enumerate(items)
    ]
    payload = {
        "schema": "taiji-u2-a5-protocol-smoke-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": "smoke_passed",
        "trainable": {
            "names": names,
            "tensors": len(names),
            "scalars": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "denominator": {"task_b": len(items), "skipped": len(skipped), "a_fit": len(a_rows)},
        "templates": contract["task_b"]["templates"],
        "template_offsets": contract["task_b"]["template_offsets"],
        "a_fit_accuracy": accuracy(a_rows),
        "b_first_pass_accuracy": accuracy(first),
        "b_counter_fact_tracking": sum(
            1
            for row, item in zip(counter_rows, items, strict=True)
            if row["output_hex"] == item["counter_target"].hex()
        )
        / len(items),
        "legal_release": legal,
        "illegal_release_refused": refusal,
        "cost": {"cpu": round(time.process_time() - cpu_start, 6)},
        "discipline": {
            "optimizer_updates": 0,
            "checkpoints_written": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
    }
    SMOKE_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "a_fit": payload["a_fit_accuracy"],
                "b_first": payload["b_first_pass_accuracy"],
                "b_counter": payload["b_counter_fact_tracking"],
                "trainable": payload["trainable"]["scalars"],
                "legal": legal,
                "refusal": refusal,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "smoke"), default="run")
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.recover is not None:
        print(recover(args.recover, contract))
        return 0
    if args.mode == "smoke":
        return smoke(contract)
    try:
        guards(contract, "run")
    except ValueError as error:
        print(json.dumps({"verdict": "refused", "reason": str(error)}))
        return 2
    try:
        return run(contract)
    except BudgetStop as error:
        if not RUN_REPORT.exists():
            RUN_REPORT.write_text(
                json.dumps(
                    {
                        "schema": "taiji-u2-a5-protocol-run-v1",
                        "card": "22.5.A.5",
                        "verdict": "budget_stop",
                        "reason": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return 3
    except Exception as error:  # noqa: BLE001
        if not RUN_REPORT.exists():
            RUN_REPORT.write_text(
                json.dumps(
                    {
                        "schema": "taiji-u2-a5-protocol-run-v1",
                        "card": "22.5.A.5",
                        "verdict": "engineering_failure",
                        "error": f"{type(error).__name__}: {error}",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
