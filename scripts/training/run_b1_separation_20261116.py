"""22.5.B.1: separate the correction update path from the emission parameters.

The A line closed with a located negative result: the only budget that produced
any correction (k=8, 8/24) cost 0.75 of the old ability, because B's updates
landed on ``byte_head`` / ``joint_readout``, which the A task also uses. Protocol
adjustments cannot fix that; the update path itself has to move.

This card is the first structural card of the B line. It adds one dedicated
plasticity target - ``correction_adapter``, four born tensors, 408 scalars,
subject identity ``taiji-u2-subject-v6`` - and freezes every emission parameter
during feedback. The adapter sits on the shared output path and is deliberately
NOT gated on record presence: on record-free A episodes it still computes, so a
badly learned adapter can damage A and the retention gate remains failable.
Freezing emission alone would have made retention tautological (A.5's trap):
without the adapter, the record channel is outside the A graph entirely.

Three properties are checked before any formal training, in smoke:

* inheritance: the 194 base tensors are bit-identical to the A.5 pre-feedback
  bundle, and the zero-initialized adapter makes v6 reproduce v4 byte for byte;
* liveness: perturbing the adapter changes the A answer digest (retention can
  fail), and the ``correction`` verification cut restores the exact v4 digest;
* gradient path: one backward probe (no optimizer step) gives the adapter and
  the record channel finite gradients while every emission tensor stays frozen.

The formal protocol is A.6's amended one (D-20261116-05/06): one error released
at a time, k=8 updates per error, no rollback, an immediate diagnostic re-ask,
and a final full pass for the gate. Thresholds are the 22.5.1 parent gate's own:
correction >= 0.80, damage <= 0.02, gain >= 0.05, world-paired 95% lower bound
> 0, three seeds, counter-fact arm, knockout arm, control arm, byte-exact
recovery. A positive result signs mechanism reachability only - not a parent,
not a capability, not check2, not 22.6.

Modes: ``smoke`` (zero optimizer steps), ``price`` (unit pricing on a discarded
model), ``run`` (formal, requires frozen contract + training authority + measured
caps), ``recover`` (independent readback), ``verify`` (re-derive gates from the
saved arms and bundles).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_b1_separation_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_b1_separation_run_20261116.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_b1_separation_smoke_20261116.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_b1_separation_price_20261116.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_b1_separation_verify_20261116.json"


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
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


def adapter_names(contract: dict) -> list[str]:
    return sorted(contract["adapter_parameters"])


def build_b1(contract: dict, seed: int):
    """v6 candidate inheriting the A.5 pre-feedback bundle bit for bit."""
    candidate = contract["candidate"]
    encoder = ROOT / candidate["encoder_bundle"]
    if hashlib.sha256(encoder.read_bytes()).hexdigest() != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    torch.manual_seed(int(seed))
    base = make_model("native", int(seed))
    base.encoder["rgb"].load_state_dict(torch.load(encoder, weights_only=False)["encoder"])
    model = base.correction_candidate()
    start = torch.load(ROOT / contract["start_bundles"][str(seed)], weights_only=False)
    missing, unexpected = model.load_state_dict(start["model"], strict=False)
    if unexpected or set(missing) != set(adapter_names(contract)):
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


def build_v4_reference(contract: dict, seed: int):
    """The same start bundle on the unmodified v4 graph, for digest equality."""
    override = {
        **contract,
        "candidate": {
            **contract["candidate"],
            "seed": int(seed),
            "parameters": 22742,
            "tensors": 194,
        },
    }
    model = a5.build_subject(override)
    model.load_state_dict(
        torch.load(ROOT / contract["start_bundles"][str(seed)], weights_only=False)["model"]
    )
    model.eval()
    return model


def set_trainable_b1(model, contract: dict) -> list[str]:
    declared = sorted([*contract["record_parameters"], *contract["adapter_parameters"]])
    found = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in set(declared))
        if name in set(declared):
            found.append(name)
    if sorted(found) != declared:
        raise ValueError("trainable_set_mismatch")
    tensors = sum(1 for p in model.parameters() if p.requires_grad)
    scalars = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if tensors != int(contract["trainable_tensor_count"]):
        raise ValueError("trainable_tensor_mismatch")
    if scalars != int(contract["trainable_scalar_count"]):
        raise ValueError("trainable_scalar_mismatch")
    for name, parameter in model.named_parameters():
        if name in set(contract["emission_parameters"]) and parameter.requires_grad:
            raise ValueError(f"emission_not_frozen:{name}")
    return sorted(found)


def codes_for(contract: dict, model, data: dict) -> dict:
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    return a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)


def a_rows(model, data: dict, codes: dict) -> list[dict]:
    return [a5.a_round(model, example, codes[example["id"]]) for example in data["fit"]]


def template_of_factory(contract: dict):
    count = len(contract["task_b"]["templates"])
    return lambda index: index % count


def purity_check(model, start: dict, contract: dict) -> dict:
    """Every non-trainable base tensor must be bit-identical to the start."""
    trainable = set(contract["record_parameters"]) | set(contract["adapter_parameters"])
    current = dict(model.named_parameters())
    drifted = []
    checked = 0
    for name, value in start["model"].items():
        if name in trainable:
            continue
        checked += 1
        if not torch.equal(current[name].detach(), value):
            drifted.append(name)
    return {"checked": checked, "drifted": drifted, "pure": not drifted}


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
    template_of = template_of_factory(contract)
    seeds = [int(value) for value in contract["plan"]["seeds"]]
    per_seed = []
    for seed in seeds:
        codes = codes_for(contract, build_v4_reference(contract, seed), data)
        v4 = build_v4_reference(contract, seed)
        v4_a = a_rows(v4, data, codes)
        v4_b = a5.b_pass(v4, items, codes, template_of)
        model, start = build_b1(contract, seed)
        last = model.correction_adapter.last
        if bool((last.weight == 0).all() and (last.bias == 0).all()):
            birth_neutral = True
        else:
            raise ValueError("adapter_not_zero_at_birth")
        v6_a = a_rows(model, data, codes)
        v6_b = a5.b_pass(model, items, codes, template_of)
        inheritance = {
            "a_digest_equal": a5.rows_digest(v6_a) == a5.rows_digest(v4_a),
            "b_digest_equal": a5.rows_digest(v6_b) == a5.rows_digest(v4_b),
            "base_tensors_bit_exact": all(
                torch.equal(dict(model.named_parameters())[name], value)
                for name, value in start["model"].items()
            ),
        }
        if not all(inheritance.values()):
            raise ValueError(f"birth_not_neutral:{seed}")
        model._verification_cuts = frozenset({"correction"})
        knock_a = a_rows(model, data, codes)
        model._verification_cuts = frozenset()
        knockout_restores_v4 = a5.rows_digest(knock_a) == a5.rows_digest(v4_a)
        if not knockout_restores_v4:
            raise ValueError(f"knockout_not_exact:{seed}")
        saved = last.weight.detach().clone()

        def probe_context(target, model=model, codes=codes):
            example = data["fit"][0]
            public = a5.p4_2.column_public(example)
            state = a5.episode(model, public, example["id"], None, False)
            with torch.no_grad(), a2.reading_code(target, codes[example["id"]]):
                _, context, _ = target.output(
                    state, target.language_goal(state), state["prediction_action"]
                )
            return context.clone()

        ctx_before = probe_context(model)
        with torch.no_grad():
            last.weight.add_(0.05)
        ctx_small = probe_context(model)
        numerical_liveness = float((ctx_small - ctx_before).abs().max())
        with torch.no_grad():
            last.weight.add_(1.95)
        perturbed_a = a_rows(model, data, codes)
        with torch.no_grad():
            last.weight.copy_(saved)
        behavioral_liveness = a5.rows_digest(perturbed_a) != a5.rows_digest(v4_a)
        if not (numerical_liveness > 0 and behavioral_liveness):
            raise ValueError(f"adapter_not_live_on_a:{seed}")
        restored = a_rows(model, data, codes)
        if a5.rows_digest(restored) != a5.rows_digest(v4_a):
            raise ValueError(f"liveness_probe_not_restored:{seed}")
        # One backward probe, zero optimizer steps: the adapter and the record
        # channel must receive gradient; emission must not even be in the graph.
        set_trainable_b1(model, contract)
        item = items[0]
        public = a5.p4_2.column_public(item["example"])
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        loss = a5.record_loss(
            model,
            public,
            item["example"]["id"],
            item["records"][0],
            item["target"],
            codes[item["example"]["id"]],
        )
        loss.backward()
        grads = dict(model.named_parameters())
        adapter_grad = float(grads["correction_adapter.last.weight"].grad.abs().sum())
        record_grad = float(grads["record_output.weight"].grad.abs().sum())
        emission_grads = [
            name for name in contract["emission_parameters"] if grads[name].grad is not None
        ]
        finite = all(
            bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
        )
        model.zero_grad(set_to_none=True)
        unchanged = all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
        if not (adapter_grad > 0 and finite and not emission_grads and unchanged):
            raise ValueError(f"gradient_path_invalid:{seed}")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # Legal release and its refusal, unchanged from A.5.
        record = a5.record_of(item["records"][0], "a5-fact")
        legal_state = a5.episode(model, public, item["example"]["id"], [record], True)
        refusal = ""
        try:
            a5.episode(
                model,
                public,
                item["example"]["id"],
                [record],
                True,
                release_ids=("a5-missing",),
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
                "birth_neutral": birth_neutral,
                "knockout_restores_v4": knockout_restores_v4,
                "retention_failable_liveness": {
                    "numerical_context_delta_at_0.05": numerical_liveness,
                    "behavioral_digest_change_at_2.0": behavioral_liveness,
                },
                "gradient_probe": {
                    "adapter_last_weight_grad_abs_sum": adapter_grad,
                    "record_output_weight_grad_abs_sum": record_grad,
                    "emission_grads_present": emission_grads,
                    "finite": finite,
                    "parameters_unchanged": unchanged,
                    "optimizer_steps": 0,
                },
                "legal_release": list(legal_state["released_feedback_records"]),
                "first_pass_correct": sum(int(row["correct"]) for row in v6_b),
                "a_baseline_correct": sum(int(row["correct"]) for row in v6_a),
            }
        )
    payload = {
        "schema": "taiji-u2-b1-separation-smoke-v1",
        "card": "22.5.B.1",
        "date": "2026-11-16",
        "seeds": per_seed,
        "skipped_unknown": len(skipped),
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
    model, _ = build_b1(contract, seed)
    codes = codes_for(contract, model, data)
    set_trainable_b1(model, contract)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    template_of = template_of_factory(contract)
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
    updates = seeds * errors * k
    # Per seed: first/final/counter/knockout B passes (4x24), diagnostic re-asks
    # (24, full option only), control pass (24), control exposure (24 episodes,
    # priced conservatively as b_rounds), baseline/final/knockout A (3x32),
    # recovery readback (32 A + 24 B), plus build/codes/save overhead.
    b_full = errors * (4 + 1 + 1 + 1) + errors
    b_lean = errors * (4 + 1) + 3
    a_per_seed = 32 * 3 + 32
    overhead = 25.0
    options = []
    for name, b_per_seed in (("full_3seed", b_full), ("lean_3seed", b_lean)):
        estimate = seeds * (
            errors * k * units["record_update"]
            + b_per_seed * units["b_round"]
            + a_per_seed * units["a_round"]
            + overhead
        )
        cap_cpu = round(estimate * (1 + float(plan["contingency"])), 3)
        options.append(
            {
                "name": name,
                "seeds": seeds,
                "k": k,
                "updates": updates,
                "diagnostic_reasks": name == "full_3seed",
                "estimate_cpu": round(estimate, 3),
                "cap_cpu": cap_cpu,
                "cap_wall": round(cap_cpu * 2, 3),
                "over_design_cap": cap_cpu > float(plan["design_cap_cpu"]),
            }
        )
    payload = {
        "schema": "taiji-u2-b1-separation-price-v1",
        "card": "22.5.B.1",
        "date": "2026-11-16",
        "unit_cpu_measured": units,
        "pricing_optimizer_steps": 8,
        "pricing_model_discarded": True,
        "options": options,
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
    }
    PRICE_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"units": units, "options": options}, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------- run


def learned_arm(
    contract: dict, seed: int, codes: dict, items: list, data: dict, spend, root: Path
) -> dict:
    model, start = build_b1(contract, seed)
    set_trainable_b1(model, contract)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
    )
    template_of = template_of_factory(contract)
    first = a5.b_pass(model, items, codes, template_of)
    baseline = a_rows(model, data, codes)
    errors = [item for item, row in zip(items, first, strict=True) if not row["correct"]]
    error_templates = [template_of(index) for index, row in enumerate(first) if not row["correct"]]
    if len(errors) < 4:
        raise ValueError("not_identifiable_by_design")
    k = int(contract["plan"]["k"])
    updates = 0
    trace = []
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
        if contract["plan"].get("diagnostic_reasks", True):
            again = a5.b_round(model, item, code, template)
            trace.append(
                {
                    "id": item["example"]["id"],
                    "template": template,
                    "immediate_correct": again["correct"],
                }
            )
    final = a5.b_pass(model, errors, codes, lambda position: error_templates[position])
    final_a = a_rows(model, data, codes)
    counter = [
        a5.b_round(model, item, codes[item["example"]["id"]], template, counter=True)
        for item, template in zip(errors, error_templates, strict=True)
    ]
    model._verification_cuts = frozenset({"correction"})
    knock_b = a5.b_pass(model, errors, codes, lambda position: error_templates[position])
    knock_a = a_rows(model, data, codes)
    model._verification_cuts = frozenset()
    purity = purity_check(model, start, contract)
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
        "damage": round(a5.accuracy(baseline) - a5.accuracy(final_a), 10),
        "counter_follow_accuracy": a5.accuracy(counter),
        "knockout_b_accuracy": a5.accuracy(knock_b),
        "knockout_a_digest": a5.rows_digest(knock_a),
        "knockout_restores_baseline": a5.rows_digest(knock_a) == a5.rows_digest(baseline),
        "purity": purity,
        "immediate_trace": trace,
        "bundle_sha256": bundle_sha,
        "rows": {
            "first": first,
            "final": final,
            "a_baseline": baseline,
            "a_final": final_a,
            "counter": counter,
            "knockout_b": knock_b,
        },
    }


def control_arm(contract: dict, seed: int, codes: dict, items: list, spend) -> dict:
    """Same exposure, zero updates: parameters and answers must not move."""
    model, start = build_b1(contract, seed)
    template_of = template_of_factory(contract)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    exposure = int(contract["plan"].get("control_exposure_episodes", len(items)))
    with torch.no_grad():
        for index, item in enumerate(items[:exposure]):
            spend()
            public = a5.p4_2.column_public(item["example"])
            record = a5.record_of(item["records"][template_of(index)], "a5-fact")
            a5.episode(model, public, item["example"]["id"], [record], True)
    unchanged = all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
    rows = a5.b_pass(model, items, codes, template_of)
    return {
        "seed": seed,
        "updates": 0,
        "exposure_episodes": exposure,
        "parameters_unchanged": unchanged,
        "second_pass_accuracy": a5.accuracy(rows),
        "second_pass_digest": a5.rows_digest(rows),
        "rows": rows,
    }


def bootstrap_lower_bound(pairs: list[tuple[int, int]], contract: dict) -> dict:
    count = int(contract["plan"]["bootstrap_resamples"])
    generator = random.Random(int(contract["plan"]["bootstrap_seed"]))
    differences = [learned - control for learned, control in pairs]
    means = []
    for _ in range(count):
        sample = [differences[generator.randrange(len(differences))] for _ in differences]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "pairs": len(pairs),
        "lower_bound_95": means[int(0.025 * count)],
        "mean": sum(differences) / len(differences),
    }


def recover_rounds(path: Path) -> dict:
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
            raise BudgetStop("b1_budget_stop")
        if time.perf_counter() - wall_start > float(contract["plan"]["cap_wall"]):
            raise BudgetStop("b1_wall_stop")

    seeds = [int(value) for value in contract["plan"]["seeds"]]
    learned, controls = [], []
    for seed in seeds:
        spend()
        codes = codes_for(contract, build_v4_reference(contract, seed), data)
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
            control = control_arm(contract, seed, codes, items, spend)
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
    lower = bootstrap_lower_bound(pairs, contract)
    rates = [entry["second_pass_accuracy"] for entry in learned]
    damages = [entry["damage"] for entry in learned]
    knockout_ok = all(
        entry["knockout_restores_baseline"]
        and entry["knockout_b_accuracy"] <= entry["first_pass_accuracy"] + zero_band
        for entry in learned
    )
    purity_ok = all(entry["purity"]["pure"] for entry in learned)
    recovery_ok = all(
        entry["recovery"]["a_digest_equal"] and entry["recovery"]["b_digest_equal"]
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
    if any(damage > retention_gate for damage in damages):
        verdict = "retention_broken_under_separation"
    elif not knockout_ok:
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
        verdict = "separation_earns_joint_gate"
    elif max(rates) <= zero_band:
        verdict = "correction_absent_under_separation"
    else:
        verdict = "correction_partial_under_separation"

    payload = {
        "schema": "taiji-u2-b1-separation-run-v1",
        "card": "22.5.B.1",
        "date": "2026-11-16",
        "verdict": verdict,
        "seeds": seeds,
        "denominator": {
            "task_b_examples": len(items),
            "first_pass_errors": [entry["first_pass_errors"] for entry in learned],
            "skipped_unknown": len(skipped),
        },
        "correction_rates": rates,
        "damages": damages,
        "gains": gains,
        "lower_bound": lower,
        "counter_follow": [entry["counter_follow_accuracy"] for entry in learned],
        "knockout_b": [entry["knockout_b_accuracy"] for entry in learned],
        "knockout_ok": knockout_ok,
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
                "damages": damages,
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
    model, _ = build_b1(contract, seed)
    model.load_state_dict(bundle["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    codes = codes_for(contract, model, data)
    items, _ = a5.task_b_examples(data, contract)
    template_of = template_of_factory(contract)
    a = a_rows(model, data, codes)
    b = a5.b_pass(model, items, codes, template_of)
    print(
        json.dumps(
            {
                "a_digest": a5.rows_digest(a),
                "b_digest": a5.rows_digest(b),
                "digest": a5.rows_digest(a + b),
            },
            ensure_ascii=False,
        )
    )
    return 0


def verify(contract: dict) -> int:
    """Re-derive every gate from the saved arms and bundles, in this process."""
    guards(contract, "verify")
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    root = ROOT / contract["run_root"]
    report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    data = atomic_load(ROOT / contract["data_root"])
    items, _ = a5.task_b_examples(data, contract)
    checks = []
    for seed in [int(value) for value in contract["plan"]["seeds"]]:
        entry = json.loads((root / f"seed-{seed}-learned.json").read_text(encoding="utf-8"))
        bundle = torch.load(root / f"seed-{seed}" / "bundle-post-feedback.pt", weights_only=False)
        model, start = build_b1(contract, seed)
        model.load_state_dict(bundle["model"])
        model.eval()
        codes = codes_for(contract, model, data)
        purity = purity_check(model, start, contract)
        checks.append({"check": f"purity_seed_{seed}", "ok": purity["pure"]})
        a = a_rows(model, data, codes)
        template_of = template_of_factory(contract)
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
        checks.append(
            {"check": f"a_digest_seed_{seed}", "ok": a5.rows_digest(a) == entry["a_fit_digest"]}
        )
        checks.append(
            {
                "check": f"b_digest_seed_{seed}",
                "ok": a5.rows_digest(b) == entry["second_pass_digest"],
            }
        )
        model._verification_cuts = frozenset({"correction"})
        knock_a = a_rows(model, data, codes)
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
                and recovery["b_digest"] == entry["second_pass_digest"],
            }
        )
    rates = [
        entry["second_pass_accuracy"]
        for entry in (
            json.loads((root / f"seed-{s}-learned.json").read_text(encoding="utf-8"))
            for s in contract["plan"]["seeds"]
        )
    ]
    checks.append({"check": "rates_match_report", "ok": rates == report["correction_rates"]})
    payload = {
        "schema": "taiji-u2-b1-separation-verify-v1",
        "card": "22.5.B.1",
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
