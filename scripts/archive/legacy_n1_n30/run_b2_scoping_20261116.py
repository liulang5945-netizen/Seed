"""22.5.B.2: scope the correction plasticity to record presence.

B.1 separated the update target (purity 176/176, knockout exact) but the
adapter learned a record-blind global context shift: damage 0.5 on two seeds,
correction 0/0/2, counter-fact 0. The attribution probe showed the presence
signal is large in the record readout (1.3-18x separability) yet ignored
(record-input contribution 1-3%). Capacity and dose are ruled out; the gap is
functional scoping.

B.2's single structural variable over B.1: the adapter delta is multiplied by
record presence (``correction="v2"``, identity ``taiji-u2-subject-v7``, the copy
branch's own ``valid`` signal). Without records the delta is exactly zero, so
the record-free A check is structural and is declared monitor-only. Retention
is instead earned on a failable arm: the 32 fit questions re-asked with a
legally released **distractor** record about a DIFFERENT entity, where the
correct answer stays the frozen truth. The adapter is live on those rounds and
can damage them.

Everything else is B.1 lean, byte for byte: same start bundles (A.5
pre-feedback), same trainable set (adapter + record channel, 22/1170), emission
frozen, k=8 no rollback, final-pass gate, counter-fact arm, knockout arm,
control arm, purity, independent-process recovery, three seeds, 22.5.1
thresholds. Training episodes are B only - distractors are never trained on,
so neutrality to irrelevant records must generalize, not be memorized.

Modes: smoke (zero optimizer steps), price (discarded model), run (authorized),
recover (independent readback), verify (re-derive from arms and bundles).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from scripts.training import run_b1_separation_20261116 as b1
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_b2_scoping_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_b2_scoping_run_20261116.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_b2_scoping_smoke_20261116.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_b2_scoping_price_20261116.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_b2_scoping_verify_20261116.json"
DISTRACTOR_ID = "b2-distractor"


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("b1_runner", ROOT / "scripts/training/run_b1_separation_20261116.py"),
        ("a5_runner", ROOT / "scripts/training/run_a5_22_5.py"),
        ("a2_runner", ROOT / "scripts/training/run_a2_byte_20261115.py"),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("events", ROOT / "taiji/u2_events.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("columns", ROOT / "taiji/u2_observation_columns.py"),
    ):
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    for key, path in contract["start_bundles"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != contract["start_bundle_sha256"][key]:
            raise ValueError(f"start_bundle_mismatch:{key}")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract["plan"].get("cap_cpu") or not contract["plan"].get("cap_wall"):
            raise ValueError("missing_measured_cap")


def build_b2(contract: dict, seed: int):
    candidate = contract["candidate"]
    encoder = ROOT / candidate["encoder_bundle"]
    if hashlib.sha256(encoder.read_bytes()).hexdigest() != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    torch.manual_seed(int(seed))
    base = make_model("native", int(seed))
    base.encoder["rgb"].load_state_dict(torch.load(encoder, weights_only=False)["encoder"])
    model = base.correction_candidate("v2")
    start = torch.load(ROOT / contract["start_bundles"][str(seed)], weights_only=False)
    missing, unexpected = model.load_state_dict(start["model"], strict=False)
    if unexpected or set(missing) != set(b1.adapter_names(contract)):
        raise ValueError(f"inheritance_mismatch:{missing}:{unexpected}")
    if sum(p.numel() for p in model.parameters()) != int(candidate["parameters"]):
        raise ValueError("candidate_parameter_count_mismatch")
    if len(list(model.named_parameters())) != int(candidate["tensors"]):
        raise ValueError("candidate_tensor_count_mismatch")
    model.parameter_revision = int(start.get("revision", 0)) + 1
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, start


def distractor_items(data: dict, contract: dict) -> list[dict]:
    """Frozen rule: a record about a DIFFERENT entity, answer stays the truth.

    Other entity = the next fit example's entity (skip one if equal); label =
    the queried example's revised position (or 左边 for unknown targets);
    template = index % 4. Single record, legally released at evaluation.
    """
    fit = data["fit"]
    order = a5.LABELS[:3]
    out = []
    for index, example in enumerate(fit):
        public = a5.p4_2.column_public(example)
        entity = public["request"].decode()[: -len(a5.SUFFIX)]
        other_index = (index + 1) % len(fit)
        other_public = a5.p4_2.column_public(fit[other_index])
        other = other_public["request"].decode()[: -len(a5.SUFFIX)]
        if other == entity:
            other_index = (index + 2) % len(fit)
            other_public = a5.p4_2.column_public(fit[other_index])
            other = other_public["request"].decode()[: -len(a5.SUFFIX)]
        if other == entity:
            raise ValueError("distractor_entity_not_distinct")
        truth = example["target"].decode()
        label = order[(order.index(truth) + 1) % 3] if truth in order else order[0]
        template = index % len(contract["task_b"]["templates"])
        raw = contract["task_b"]["templates"][template].format(entity=other, label=label).encode()
        out.append({"example": example, "raw": raw, "template": template, "other_entity": other})
    return out


def distractor_round(model, item: dict, code: torch.Tensor) -> dict:
    public = a5.p4_2.column_public(item["example"])
    record = a5.record_of(item["raw"], DISTRACTOR_ID)
    state = a5.episode(model, public, item["example"]["id"], [record], True)
    got = a5.answer_bytes(model, state, code)
    target = item["example"]["target"]
    return {
        "id": item["example"]["id"],
        "template": item["template"],
        "other_entity": item["other_entity"],
        "target": target.decode(),
        "output_hex": got["raw"].hex(),
        "eos": got["eos"],
        "valid_utf8": got["valid_utf8"],
        "correct": bool(got["eos"] and got["valid_utf8"] and got["raw"] == target),
    }


def gate_structure_check(model, data: dict, items: list, codes: dict, contract: dict) -> dict:
    """Without records the returned context equals the cut context bit for bit;
    with a released record the delta equals the ungated adapter output."""
    example = data["fit"][0]
    public = a5.p4_2.column_public(example)
    code = codes[example["id"]]
    state = a5.episode(model, public, example["id"], None, False)
    goal = model.language_goal(state)
    with torch.no_grad(), a2.reading_code(model, code):
        _, ctx_a, _ = model.output(state, goal, state["prediction_action"])
        model._verification_cuts = frozenset({"correction"})
        _, ctx_cut, _ = model.output(state, goal, state["prediction_action"])
        model._verification_cuts = frozenset()
    closed_exact = torch.equal(ctx_a, ctx_cut)
    item = items[0]
    record = a5.record_of(item["records"][0], "a5-fact")
    state_b = a5.episode(model, public, item["example"]["id"], [record], True)
    with torch.no_grad(), a2.reading_code(model, code):
        base_ctx = model.branch_context(state_b, goal, state_b["prediction_action"])
        read, *_ = model.record_read(
            state_b,
            goal,
            state_b["previous_byte"],
            model.read_h(state_b, goal)[:, None],
        )
        read = read[:, 0]
        _, ctx_b, _ = model.output(state_b, goal, state_b["prediction_action"])
        manual = base_ctx + model.correction_adapter(torch.cat([read, base_ctx], -1))
    open_matches_ungated = torch.equal(ctx_b, manual)
    return {
        "closed_exact_zero": bool(closed_exact),
        "open_equals_ungated": bool(open_matches_ungated),
    }


# ------------------------------------------------------------------------- smoke


def smoke(contract: dict) -> int:
    guards(contract, "smoke")
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    data = atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    items, skipped = a5.task_b_examples(data, contract)
    distractors = distractor_items(data, contract)
    template_of = b1.template_of_factory(contract)
    per_seed = []
    for seed in [int(value) for value in contract["plan"]["seeds"]]:
        codes = b1.codes_for(contract, b1.build_v4_reference(contract, seed), data)
        v4 = b1.build_v4_reference(contract, seed)
        v4_a = b1.a_rows(v4, data, codes)
        v4_b = a5.b_pass(v4, items, codes, template_of)
        model, start = build_b2(contract, seed)
        last = model.correction_adapter.last
        if not bool((last.weight == 0).all() and (last.bias == 0).all()):
            raise ValueError("adapter_not_zero_at_birth")
        v7_a = b1.a_rows(model, data, codes)
        v7_b = a5.b_pass(model, items, codes, template_of)
        inheritance = {
            "a_digest_equal": a5.rows_digest(v7_a) == a5.rows_digest(v4_a),
            "b_digest_equal": a5.rows_digest(v7_b) == a5.rows_digest(v4_b),
            "base_tensors_bit_exact": all(
                torch.equal(dict(model.named_parameters())[name], value)
                for name, value in start["model"].items()
            ),
        }
        if not all(inheritance.values()):
            raise ValueError(f"birth_not_neutral:{seed}")
        # Distractor rounds at birth equal the v4-with-record digest (gate
        # open, adapter zero). Measured BEFORE any perturbation.
        v4_distractor = [
            distractor_round(v4, item, codes[item["example"]["id"]]) for item in distractors
        ]
        birth_distractor = [
            distractor_round(model, item, codes[item["example"]["id"]]) for item in distractors
        ]
        distractor_birth_neutral = a5.rows_digest(birth_distractor) == a5.rows_digest(v4_distractor)
        if not distractor_birth_neutral:
            raise ValueError(f"distractor_birth_not_neutral:{seed}")
        # Gate closed: perturbing the adapter must NOT move record-free A.
        saved = last.weight.detach().clone()
        with torch.no_grad():
            last.weight.add_(2.0)
        perturbed_a = b1.a_rows(model, data, codes)
        closed_under_perturbation = a5.rows_digest(perturbed_a) == a5.rows_digest(v4_a)
        if not closed_under_perturbation:
            raise ValueError(f"gate_not_closed:{seed}")
        # Gate open: the same perturbation MUST move record-present B rounds.
        perturbed_b = a5.b_pass(model, items, codes, template_of)
        open_under_perturbation = a5.rows_digest(perturbed_b) != a5.rows_digest(v4_b)
        if not open_under_perturbation:
            raise ValueError(f"gate_not_open:{seed}")
        with torch.no_grad():
            last.weight.copy_(saved)
        # Knockout restores v4 even with records present.
        model._verification_cuts = frozenset({"correction"})
        with torch.no_grad():
            last.weight.add_(2.0)
        knock_b = a5.b_pass(model, items, codes, template_of)
        with torch.no_grad():
            last.weight.copy_(saved)
        model._verification_cuts = frozenset()
        knockout_exact = a5.rows_digest(knock_b) == a5.rows_digest(v4_b)
        if not knockout_exact:
            raise ValueError(f"knockout_not_exact:{seed}")
        # Gradient probe: zero optimizer steps; adapter gradient is zero on a
        # record-free episode and nonzero on a released-record episode.
        b1.set_trainable_b1(model, contract)
        item = items[0]
        public = a5.p4_2.column_public(item["example"])
        code = codes[item["example"]["id"]]
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        # record_loss releases a record, so this probes the OPEN gate: the
        # adapter must receive gradient. The closed gate was already proven
        # structurally above (perturbation cannot move record-free A).
        loss_open = a5.record_loss(
            model, public, item["example"]["id"], item["records"][0], item["target"], code
        )
        loss_open.backward()
        grad_open = float(model.correction_adapter.last.weight.grad.abs().sum())
        model.zero_grad(set_to_none=True)
        unchanged = all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
        if not (grad_open > 0 and unchanged):
            raise ValueError(f"gradient_path_invalid:{seed}")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        gate = gate_structure_check(model, data, items, codes, contract)
        if not (gate["closed_exact_zero"] and gate["open_equals_ungated"]):
            raise ValueError(f"gate_structure_invalid:{seed}")
        record = a5.record_of(item["records"][0], "a5-fact")
        legal_state = a5.episode(model, public, item["example"]["id"], [record], True)
        refusal = ""
        try:
            a5.episode(
                model, public, item["example"]["id"], [record], True, release_ids=("a5-missing",)
            )
            refusal = "accepted"
        except ValueError as error:
            refusal = str(error)
        if "not legally released" not in refusal:
            raise ValueError(f"illegal_release_not_refused:{seed}")
        per_seed.append(
            {
                "seed": seed,
                "inheritance": inheritance,
                "gate_closed_under_perturbation": closed_under_perturbation,
                "gate_open_under_perturbation": open_under_perturbation,
                "distractor_birth_neutral": distractor_birth_neutral,
                "distractor_birth_accuracy": a5.accuracy(birth_distractor),
                "knockout_restores_v4_with_records": knockout_exact,
                "gradient_probe": {
                    "adapter_grad_abs_sum_open": grad_open,
                    "parameters_unchanged": unchanged,
                    "optimizer_steps": 0,
                },
                "gate_structure": gate,
                "legal_release": list(legal_state["released_feedback_records"]),
                "first_pass_correct": sum(int(row["correct"]) for row in v7_b),
                "a_baseline_correct": sum(int(row["correct"]) for row in v7_a),
            }
        )
    payload = {
        "schema": "taiji-u2-b2-scoping-smoke-v1",
        "card": "22.5.B.2",
        "date": "2026-11-16",
        "seeds": per_seed,
        "skipped_unknown": len(skipped),
        "distractor_denominator": len(distractors),
        "optimizer_steps": 0,
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
    }
    SMOKE_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"verdict": "smoke_passed", "cost": payload["cost"]}, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------------- price


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    data = atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    items, _ = a5.task_b_examples(data, contract)
    seed = int(contract["plan"]["seeds"][0])
    model, _ = build_b2(contract, seed)
    codes = b1.codes_for(contract, model, data)
    b1.set_trainable_b1(model, contract)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    template_of = b1.template_of_factory(contract)
    item = items[0]
    public = a5.p4_2.column_public(item["example"])
    code = codes[item["example"]["id"]]
    update_times, b_times, a_times = [], [], []
    for _ in range(8):
        started = time.process_time()
        optimizer.zero_grad(set_to_none=True)
        loss = a5.record_loss(
            model, public, item["example"]["id"], item["records"][0], item["target"], code
        )
        loss.backward()
        optimizer.step()
        model.parameter_revision += 1
        update_times.append(time.process_time() - started)
    for index in range(8):
        started = time.process_time()
        a5.b_round(model, items[index], codes[items[index]["example"]["id"]], template_of(index))
        b_times.append(time.process_time() - started)
    for index in range(8):
        example = data["fit"][index]
        started = time.process_time()
        a5.a_round(model, example, codes[example["id"]])
        a_times.append(time.process_time() - started)
    units = {
        "record_update": round(statistics.median(update_times), 6),
        "b_round": round(statistics.median(b_times), 6),
        "a_round": round(statistics.median(a_times), 6),
    }
    plan = contract["plan"]
    k = int(plan["k"])
    seeds = len(plan["seeds"])
    errors = int(plan["errors_per_seed"])
    distractor_rounds = int(plan["distractor_rounds_per_seed"])
    updates = seeds * errors * k
    # Per seed: first/final/counter/knockout/control B passes (5x24) + 3 control
    # exposures + 32 distractor rounds (priced as b_rounds), baseline/final/
    # knockout/recovery A (4x32), plus build/codes/save overhead.
    b_per_seed = errors * 5 + 3 + distractor_rounds
    a_per_seed = 32 * 4
    overhead = 25.0
    estimate = seeds * (
        errors * k * units["record_update"]
        + b_per_seed * units["b_round"]
        + a_per_seed * units["a_round"]
        + overhead
    )
    cap_cpu = round(estimate * (1 + float(plan["contingency"])), 3)
    option = {
        "name": "scoping_3seed",
        "seeds": seeds,
        "k": k,
        "updates": updates,
        "distractor_rounds": seeds * distractor_rounds,
        "estimate_cpu": round(estimate, 3),
        "cap_cpu": cap_cpu,
        "cap_wall": round(cap_cpu * 2, 3),
        "over_design_cap": cap_cpu > float(plan["design_cap_cpu"]),
    }
    payload = {
        "schema": "taiji-u2-b2-scoping-price-v1",
        "card": "22.5.B.2",
        "date": "2026-11-16",
        "unit_cpu_measured": units,
        "pricing_optimizer_steps": 8,
        "pricing_model_discarded": True,
        "options": [option],
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
    }
    PRICE_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"units": units, "option": option}, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------- run


def learned_arm(
    contract: dict, seed: int, codes: dict, items: list, data: dict, spend, root: Path
) -> dict:
    model, start = build_b2(contract, seed)
    b1.set_trainable_b1(model, contract)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    template_of = b1.template_of_factory(contract)
    first = a5.b_pass(model, items, codes, template_of)
    baseline = b1.a_rows(model, data, codes)
    distractors = distractor_items(data, contract)
    baseline_distractor = [
        distractor_round(model, item, codes[item["example"]["id"]]) for item in distractors
    ]
    errors = [item for item, row in zip(items, first, strict=True) if not row["correct"]]
    error_templates = [template_of(index) for index, row in enumerate(first) if not row["correct"]]
    if len(errors) < 4:
        raise ValueError("not_identifiable_by_design")
    k = int(contract["plan"]["k"])
    updates = 0
    for item, template in zip(errors, error_templates, strict=True):
        public = a5.p4_2.column_public(item["example"])
        code = codes[item["example"]["id"]]
        for _ in range(k):
            spend()
            optimizer.zero_grad(set_to_none=True)
            loss = a5.record_loss(
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
    final = a5.b_pass(model, errors, codes, lambda position: error_templates[position])
    final_a = b1.a_rows(model, data, codes)
    counter = [
        a5.b_round(model, item, codes[item["example"]["id"]], template, counter=True)
        for item, template in zip(errors, error_templates, strict=True)
    ]
    distractor_rows = [
        distractor_round(model, item, codes[item["example"]["id"]]) for item in distractors
    ]
    model._verification_cuts = frozenset({"correction"})
    knock_b = a5.b_pass(model, errors, codes, lambda position: error_templates[position])
    knock_a = b1.a_rows(model, data, codes)
    model._verification_cuts = frozenset()
    gate = gate_structure_check(model, data, items, codes, contract)
    purity = b1.purity_check(model, start, contract)
    bundle_path = root / f"seed-{seed}" / "bundle-post-feedback.pt"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_sha = a5.p4_2.save_checkpoint(
        bundle_path,
        {
            "model": {n: p.detach().clone() for n, p in model.state_dict().items()},
            "seed": seed,
            "k": k,
            "revision": int(model.parameter_revision),
            "start_bundle": contract["start_bundles"][str(seed)],
        },
    )
    return {
        "seed": seed,
        "updates": updates,
        "first_pass_accuracy": a5.accuracy(first),
        "first_pass_errors": len(errors),
        "baseline_a_accuracy": a5.accuracy(baseline),
        "baseline_a_digest": a5.rows_digest(baseline),
        "second_pass_accuracy": a5.accuracy(final),
        "second_pass_digest": a5.rows_digest(final),
        "a_fit_accuracy": a5.accuracy(final_a),
        "a_fit_digest": a5.rows_digest(final_a),
        "record_free_damage": round(a5.accuracy(baseline) - a5.accuracy(final_a), 10),
        "distractor_a_accuracy": a5.accuracy(distractor_rows),
        "distractor_a_digest": a5.rows_digest(distractor_rows),
        "baseline_distractor_accuracy": a5.accuracy(baseline_distractor),
        "distractor_damage": round(
            a5.accuracy(baseline_distractor) - a5.accuracy(distractor_rows), 10
        ),
        "counter_follow_accuracy": a5.accuracy(counter),
        "knockout_b_accuracy": a5.accuracy(knock_b),
        "knockout_a_digest": a5.rows_digest(knock_a),
        "knockout_restores_baseline": a5.rows_digest(knock_a) == a5.rows_digest(baseline),
        "gate_structure": gate,
        "purity": purity,
        "bundle_sha256": bundle_sha,
        "rows": {
            "first": first,
            "final": final,
            "a_baseline": baseline,
            "a_final": final_a,
            "counter": counter,
            "knockout_b": knock_b,
            "distractor": distractor_rows,
            "baseline_distractor": baseline_distractor,
        },
    }


def recover_rounds(path: Path) -> dict:
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(Path(__file__)), "--recover", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"recovery_failed:{completed.stderr[-400:]}")
    return cast(dict, json.loads(completed.stdout.strip().splitlines()[-1]))


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    root = ROOT / contract["run_root"]
    root.mkdir(parents=True, exist_ok=True)
    items, skipped = a5.task_b_examples(data, contract)

    def spend() -> None:
        if time.process_time() - cpu_start > float(contract["plan"]["cap_cpu"]):
            raise BudgetStop("b2_budget_stop")
        if time.perf_counter() - wall_start > float(contract["plan"]["cap_wall"]):
            raise BudgetStop("b2_wall_stop")

    seeds = [int(value) for value in contract["plan"]["seeds"]]
    learned, controls = [], []
    for seed in seeds:
        spend()
        codes = b1.codes_for(contract, b1.build_v4_reference(contract, seed), data)
        learned_path = root / f"seed-{seed}-learned.json"
        if learned_path.exists():
            entry = json.loads(learned_path.read_text(encoding="utf-8"))
        else:
            started = time.process_time()
            entry = learned_arm(contract, seed, codes, items, data, spend, root)
            entry["arm_cpu"] = round(time.process_time() - started, 6)
            recovery = recover_rounds(root / f"seed-{seed}" / "bundle-post-feedback.pt")
            entry["recovery"] = {
                "a_digest_equal": recovery["a_digest"] == entry["a_fit_digest"],
                "b_digest_equal": recovery["b_digest"] == entry["second_pass_digest"],
                "distractor_digest_equal": recovery["distractor_digest"]
                == entry["distractor_a_digest"],
                "digest": recovery["digest"],
            }
            learned_path.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        learned.append(entry)
        control_path = root / f"seed-{seed}-control.json"
        if control_path.exists():
            control = json.loads(control_path.read_text(encoding="utf-8"))
        else:
            control = b1.control_arm(contract, seed, codes, items, spend)
            control_path.write_text(
                json.dumps(control, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        controls.append(control)

    criteria = contract["criteria"]
    gate = float(criteria["correction_gate"])
    zero_band = float(criteria["zero_band"])
    retention_gate = float(criteria["retention_gate"])
    gains = [
        round(entry["second_pass_accuracy"] - control["second_pass_accuracy"], 10)
        for entry, control in zip(learned, controls, strict=True)
    ]
    pairs = [
        (int(row["correct"]), int(crow["correct"]))
        for entry, control in zip(learned, controls, strict=True)
        for row, crow in zip(entry["rows"]["final"], control["rows"], strict=True)
    ]
    lower = b1.bootstrap_lower_bound(pairs, contract)
    rates = [entry["second_pass_accuracy"] for entry in learned]
    distractor_damages = [entry["distractor_damage"] for entry in learned]
    knockout_ok = all(
        entry["knockout_restores_baseline"]
        and entry["knockout_b_accuracy"] <= entry["first_pass_accuracy"] + zero_band
        for entry in learned
    )
    gate_ok = all(
        entry["gate_structure"]["closed_exact_zero"]
        and entry["gate_structure"]["open_equals_ungated"]
        for entry in learned
    )
    purity_ok = all(entry["purity"]["pure"] for entry in learned)
    recovery_ok = all(
        entry["recovery"]["a_digest_equal"]
        and entry["recovery"]["b_digest_equal"]
        and entry["recovery"]["distractor_digest_equal"]
        for entry in learned
    )
    control_ok = all(control["parameters_unchanged"] for control in controls)
    counter_ok = all(
        entry["counter_follow_accuracy"] >= gate
        for entry in learned
        if entry["second_pass_accuracy"] >= gate
    )
    main_ok = all(entry["a_fit_accuracy"] >= float(criteria["main_gate"]) for entry in learned)
    stability_ok = all(
        sum(
            int(f["correct"] and not s["correct"])
            for f, s in zip(entry["rows"]["first"], entry["rows"]["final"], strict=True)
        )
        <= max(1, int(0.10 * sum(int(f["correct"]) for f in entry["rows"]["first"])))
        for entry in learned
    )
    if any(damage > retention_gate for damage in distractor_damages):
        verdict = "retention_broken_under_scoping"
    elif not knockout_ok or not gate_ok:
        verdict = "knockout_failed"
    elif not (purity_ok and recovery_ok and control_ok):
        verdict = "engineering_failure"
    elif not counter_ok:
        verdict = "counterfactual_failed"
    elif (
        all(rate >= gate for rate in rates)
        and all(gain >= float(criteria["feedback_gain_gate"]) for gain in gains)
        and lower["lower_bound_95"] > 0
        and main_ok
        and stability_ok
    ):
        verdict = "scoping_earns_joint_gate"
    elif max(rates) <= zero_band:
        verdict = "correction_absent_under_scoping"
    else:
        verdict = "correction_partial_under_scoping"

    payload = {
        "schema": "taiji-u2-b2-scoping-run-v1",
        "card": "22.5.B.2",
        "date": "2026-11-16",
        "verdict": verdict,
        "seeds": seeds,
        "denominator": {
            "task_b_examples": len(items),
            "first_pass_errors": [entry["first_pass_errors"] for entry in learned],
            "distractor_rounds": 32,
            "skipped_unknown": len(skipped),
        },
        "correction_rates": rates,
        "distractor_damages": distractor_damages,
        "distractor_accuracies": [entry["distractor_a_accuracy"] for entry in learned],
        "baseline_distractor_accuracies": [
            entry["baseline_distractor_accuracy"] for entry in learned
        ],
        "record_free_damages": [entry["record_free_damage"] for entry in learned],
        "record_free_damage_is_structural_monitor": True,
        "gains": gains,
        "lower_bound": lower,
        "counter_follow": [entry["counter_follow_accuracy"] for entry in learned],
        "knockout_b": [entry["knockout_b_accuracy"] for entry in learned],
        "knockout_ok": knockout_ok,
        "gate_ok": gate_ok,
        "purity_ok": purity_ok,
        "recovery_ok": recovery_ok,
        "control_ok": control_ok,
        "main_ok": main_ok,
        "stability_ok": stability_ok,
        "accepted_parent": False,
        "capability_claims": 0,
        "check2": "not_read",
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
    }
    RUN_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "rates": rates,
                "distractor_damages": distractor_damages,
                "lower_bound": lower["lower_bound_95"],
                "cost": payload["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def recover(path: Path) -> int:
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bundle = torch.load(path, weights_only=False)
    seed = int(bundle["seed"])
    data = atomic_load(ROOT / contract["data_root"])
    model, _ = build_b2(contract, seed)
    model.load_state_dict(bundle["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    codes = b1.codes_for(contract, model, data)
    items, _ = a5.task_b_examples(data, contract)
    template_of = b1.template_of_factory(contract)
    a = b1.a_rows(model, data, codes)
    b = a5.b_pass(model, items, codes, template_of)
    distractors = distractor_items(data, contract)
    d = [distractor_round(model, item, codes[item["example"]["id"]]) for item in distractors]
    print(
        json.dumps(
            {
                "a_digest": a5.rows_digest(a),
                "b_digest": a5.rows_digest(b),
                "distractor_digest": a5.rows_digest(d),
                "digest": a5.rows_digest(a + b + d),
            },
            ensure_ascii=False,
        )
    )
    return 0


def verify(contract: dict) -> int:
    guards(contract, "verify")
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    root = ROOT / contract["run_root"]
    report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    data = atomic_load(ROOT / contract["data_root"])
    items, _ = a5.task_b_examples(data, contract)
    distractors = distractor_items(data, contract)
    checks = []
    for seed in [int(value) for value in contract["plan"]["seeds"]]:
        entry = json.loads((root / f"seed-{seed}-learned.json").read_text(encoding="utf-8"))
        bundle = torch.load(root / f"seed-{seed}" / "bundle-post-feedback.pt", weights_only=False)
        model, start = build_b2(contract, seed)
        model.load_state_dict(bundle["model"])
        model.eval()
        codes = b1.codes_for(contract, model, data)
        purity = b1.purity_check(model, start, contract)
        checks.append({"check": f"purity_seed_{seed}", "ok": purity["pure"]})
        a = b1.a_rows(model, data, codes)
        template_of = b1.template_of_factory(contract)
        errors = [
            item
            for item, row in zip(items, entry["rows"]["first"], strict=True)
            if not row["correct"]
        ]
        templates = [
            template_of(index)
            for index, row in enumerate(entry["rows"]["first"])
            if not row["correct"]
        ]
        b = a5.b_pass(
            model, errors, codes, lambda position, templates=templates: templates[position]
        )
        d = [distractor_round(model, item, codes[item["example"]["id"]]) for item in distractors]
        checks.append(
            {"check": f"a_digest_seed_{seed}", "ok": a5.rows_digest(a) == entry["a_fit_digest"]}
        )
        checks.append(
            {
                "check": f"b_digest_seed_{seed}",
                "ok": a5.rows_digest(b) == entry["second_pass_digest"],
            }
        )
        checks.append(
            {
                "check": f"distractor_digest_seed_{seed}",
                "ok": a5.rows_digest(d) == entry["distractor_a_digest"],
            }
        )
        model._verification_cuts = frozenset({"correction"})
        knock_a = b1.a_rows(model, data, codes)
        model._verification_cuts = frozenset()
        checks.append(
            {
                "check": f"knockout_seed_{seed}",
                "ok": a5.rows_digest(knock_a) == entry["baseline_a_digest"],
            }
        )
        recovery = recover_rounds(root / f"seed-{seed}" / "bundle-post-feedback.pt")
        checks.append(
            {
                "check": f"recovery_seed_{seed}",
                "ok": recovery["a_digest"] == entry["a_fit_digest"]
                and recovery["b_digest"] == entry["second_pass_digest"]
                and recovery["distractor_digest"] == entry["distractor_a_digest"],
            }
        )
    rates = [
        json.loads((root / f"seed-{s}-learned.json").read_text(encoding="utf-8"))[
            "second_pass_accuracy"
        ]
        for s in contract["plan"]["seeds"]
    ]
    checks.append({"check": "rates_match_report", "ok": rates == report["correction_rates"]})
    payload = {
        "schema": "taiji-u2-b2-scoping-verify-v1",
        "card": "22.5.B.2",
        "date": "2026-11-16",
        "checks": checks,
        "all_ok": all(check["ok"] for check in checks),
        "cost": {"cpu": round(time.process_time() - cpu_start, 6)},
    }
    VERIFY_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"all_ok": payload["all_ok"], "checks": len(checks)}, ensure_ascii=False))
    return 0 if payload["all_ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--price", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--recover", type=str, default=None)
    args = parser.parse_args()
    if args.recover:
        return recover(Path(args.recover))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.smoke:
        return smoke(contract)
    if args.price:
        return price(contract)
    if args.run:
        return run(contract)
    if args.verify:
        return verify(contract)
    parser.error("choose a mode")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
