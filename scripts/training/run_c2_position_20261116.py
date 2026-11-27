"""22.5.C.2: position-directed retrieval (final C-line revision).

C.1 located the representation failure precisely: the value readout read is
record-invariant (cross-record std 0.026 vs norm 5.73) because position
attention is near-uniform (mass 0.157, entropy 3.66), averaging the whole record
into a constant and erasing the label signal. C.2's single variable over C.1 is
a position-focus loss that supervises the attention mass onto the label bytes,
forcing record_position_query/key to point at the answer so read carries the
label.

PRE-COMMITMENT (owner, D-20261116-15): C.2 is the second and FINAL C-line
architecture revision. If the representation gates still fail, the byte-world
joint-gate effort is globally closed out (MAINLINE section 5); no further local
additions.

Everything else inherits C.1 byte for byte: v4 record channel trainable (18
tensors), decode_head scaffold (not in output), emission frozen + purity, same
dataset, decode >= 0.95 and selectivity >= 0.95 gates on three seeds.

Modes: smoke, price, run, recover, verify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from scripts.training import run_b1_separation_20261116 as b1
from scripts.training import run_c1_retrieval_20261116 as c1
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_subject import record_inputs

CONTRACT = ROOT / "plans/manifests/taiji_u2_c2_position_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_c2_position_run_20261116.json"
SMOKE_REPORT = ROOT / "reports/taiji_u2_c2_position_smoke_20261116.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_c2_position_price_20261116.json"
VERIFY_REPORT = ROOT / "reports/taiji_u2_c2_position_verify_20261116.json"
LABELS = a5.LABELS
POSITIONS = LABELS[:3]
EPS = 1e-8


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("c1_runner", ROOT / "scripts/training/run_c1_retrieval_20261116.py"),
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


def retrieval_full(model, state, code):
    """read, route, and the position attention weights/valid for record 0."""
    goal = model.language_goal(state)
    c = model.c
    with a2.reading_code(model, code):
        queries = model.read_h(state, goal)[:, None]
        tokens, ids, valid = model.record_units(state)
        goal_units = model.text_units(state["goal_raw"])
        if goal_units.numel():
            scores = model.record_goal_query(queries) @ goal_units.T / math.sqrt(c.d)
            goal_context = scores.softmax(-1) @ goal_units
        else:
            goal_context = goal[None, None].expand(c.branches, 1, -1)
        previous_value = model.previous_embedding(state["previous_byte"])[None, None].expand(
            c.branches, 1, -1
        )
        query = model.record_query(torch.cat([queries, goal_context, previous_value], -1))
        pools = tokens.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        route_score = query @ model.record_key(pools).T / math.sqrt(c.d)
        null = model.record_null(query)
        position_query = model.record_position_query(
            torch.cat(
                [
                    query[:, 0, None].expand(-1, len(pools), -1),
                    pools[None].expand(c.branches, -1, -1),
                ],
                -1,
            )
        )
        position_score = torch.einsum(
            "brd,rpd->brp", position_query, model.record_position_key(tokens)
        ) / math.sqrt(c.d)
        position_score = (
            position_score
            - nn.functional.softplus(model.coverage_logit) * state["cursor"]["coverage"]
        )
        log_alpha, log_beta = record_inputs(
            torch.cat([route_score[:, 0], null[:, 0]], -1), position_score, valid
        )
        weights = (log_alpha[:, :-1, None] + log_beta).exp()
        context = (
            torch.einsum("brp,rpd->bd", weights, model.record_value(tokens))
            + log_alpha[:, -1:].exp() * model.record_null_value
        )
        read = model.record_output(context)
    return read, route_score[:, 0], weights[0, 0], valid[0]


def label_span(raw: bytes, label: str) -> tuple[int, int]:
    lo = raw.index(label.encode())
    return lo, lo + len(label.encode())


def position_focus_loss(weights, valid, raw, label):
    lo, hi = label_span(raw, label)
    mask = torch.zeros_like(weights)
    mask[lo:hi] = 1.0
    mass_label = (weights * mask).sum()
    mass_valid = (weights * valid.to(weights.dtype)).sum().clamp_min(EPS)
    return -torch.log(mass_label / mass_valid + EPS)


def train_step(model, head, optimizer, decode_item, pair, code, margin):
    optimizer.zero_grad(set_to_none=True)
    state = c1.decode_episode(model, decode_item, code)
    read, _, weights, valid = retrieval_full(model, state, code)
    logits = head(read)
    target = torch.tensor([LABELS.index(decode_item["label"])] * logits.shape[0])
    decode_loss = nn.functional.cross_entropy(logits, target)
    focus_loss = position_focus_loss(weights, valid, decode_item["raw"], decode_item["label"])
    rm, rd = c1.selectivity_routes(model, pair, code)
    sel_loss = torch.clamp(margin - (rm - rd), min=0.0)
    loss = decode_loss + sel_loss + focus_loss
    loss.backward()
    optimizer.step()
    model.parameter_revision += 1
    return float(decode_loss.detach()), float(sel_loss.detach()), float(focus_loss.detach())


def diagnostics(model, head, dataset, codes):
    """Answer attention mass + read record-separability on fit records."""
    masses, reads = [], []
    with torch.no_grad():
        for item in dataset["decode"]["fit"]:
            code = codes[item["example"]["id"]]
            state = c1.decode_episode(model, item, code)
            read, _, weights, valid = retrieval_full(model, state, code)
            lo, hi = label_span(item["raw"], item["label"])
            mask = torch.zeros_like(weights)
            mask[lo:hi] = 1.0
            mass = float(
                (weights * mask).sum() / (weights * valid.to(weights.dtype)).sum().clamp_min(EPS)
            )
            masses.append(mass)
            reads.append(read[0])
    R = torch.stack(reads)
    return {
        "answer_mass_mean": round(sum(masses) / len(masses), 6),
        "read_std_over_norm": round(
            float(R.std(0).mean() / R.norm(dim=1).mean().clamp_min(EPS)), 6
        ),
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
    dataset = c1.build_dataset(data, contract)
    per_seed = []
    for seed in [int(value) for value in contract["plan"]["seeds"]]:
        model = b1.build_v4_reference(contract, seed)
        start = torch.load(ROOT / contract["start_bundles"][str(seed)], weights_only=False)
        codes = b1.codes_for(contract, model, data)
        head = c1.make_decode_head(model.c.d)
        birth = c1.evaluate(model, head, dataset, codes, contract)
        diag = diagnostics(model, head, dataset, codes)
        a_start = b1.a_rows(model, data, codes)
        c1.set_trainable_c1(model, contract)
        for p in head.parameters():
            p.requires_grad_(True)
        item = dataset["decode"]["train"][0]
        code = codes[item["example"]["id"]]
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        head_saved = head.last.weight.detach().clone()
        with torch.no_grad():
            head.last.weight.add_(0.1)
        state = c1.decode_episode(model, item, code)
        read, _, weights, valid = retrieval_full(model, state, code)
        logits = head(read)
        target = torch.tensor([LABELS.index(item["label"])] * logits.shape[0])
        loss = nn.functional.cross_entropy(logits, target) + position_focus_loss(
            weights, valid, item["raw"], item["label"]
        )
        loss.backward()
        record_grad = float(cast(torch.Tensor, model.record_output.weight.grad).abs().sum())
        position_grad = float(cast(torch.Tensor, model.record_position_key.weight.grad).abs().sum())
        head_grad = float(cast(torch.Tensor, head.last.weight.grad).abs().sum())
        emission_grads = [
            n
            for n in contract["emission_parameters"]
            if dict(model.named_parameters())[n].grad is not None
        ]
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        with torch.no_grad():
            head.last.weight.copy_(head_saved)
        unchanged = all(torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
        if not (
            record_grad > 0
            and position_grad > 0
            and head_grad > 0
            and not emission_grads
            and unchanged
        ):
            raise ValueError(f"gradient_path_invalid:{seed}")
        for p in model.parameters():
            p.requires_grad_(False)
        for p in head.parameters():
            p.requires_grad_(False)
        a_after = b1.a_rows(model, data, codes)
        answer_unchanged = a5.rows_digest(a_after) == a5.rows_digest(a_start)
        purity = c1.purity_check_c1(model, start, contract)
        per_seed.append(
            {
                "seed": seed,
                "birth_decode_accuracy": birth["decode_fit_accuracy"],
                "birth_selectivity_accuracy": birth["selectivity_fit_accuracy"],
                "birth_diagnostics": diag,
                "gradient_probe": {
                    "record_output_grad": record_grad,
                    "record_position_key_grad": position_grad,
                    "head_last_grad": head_grad,
                    "emission_grads_present": emission_grads,
                    "parameters_unchanged": unchanged,
                    "optimizer_steps": 0,
                },
                "answer_path_bit_identical": answer_unchanged,
                "purity": purity,
            }
        )
    payload = {
        "schema": "taiji-u2-c2-position-smoke-v1",
        "card": "22.5.C.2",
        "date": "2026-11-16",
        "seeds": per_seed,
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
    dataset = c1.build_dataset(data, contract)
    seed = int(contract["plan"]["seeds"][0])
    model = b1.build_v4_reference(contract, seed)
    codes = b1.codes_for(contract, model, data)
    head = c1.make_decode_head(model.c.d)
    c1.set_trainable_c1(model, contract)
    for p in head.parameters():
        p.requires_grad_(True)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
        contract["feedback_optimizer"],
    )
    margin = float(contract["plan"]["selectivity_margin"])
    times = []
    for index in range(8):
        item = dataset["decode"]["train"][index]
        pair = dataset["pairs"][index % len(dataset["pairs"])]
        code = codes[item["example"]["id"]]
        started = time.process_time()
        train_step(model, head, optimizer, item, pair, code, margin)
        times.append(time.process_time() - started)
    unit = round(statistics.median(times), 6)
    plan = contract["plan"]
    seeds = len(plan["seeds"])
    updates = int(plan["updates_per_seed"])
    eval_per_seed = len(dataset["decode"]["fit"]) + 2 * len(dataset["pairs"])
    estimate = seeds * (updates * unit + eval_per_seed * unit * 0.5 + 25.0)
    cap_cpu = round(estimate * (1 + float(plan["contingency"])), 3)
    option = {
        "name": "position_3seed",
        "seeds": seeds,
        "updates_per_seed": updates,
        "updates_total": seeds * updates,
        "unit_cpu_measured": unit,
        "estimate_cpu": round(estimate, 3),
        "cap_cpu": cap_cpu,
        "cap_wall": round(cap_cpu * 2, 3),
        "over_design_cap": cap_cpu > float(plan["design_cap_cpu"]),
    }
    payload = {
        "schema": "taiji-u2-c2-position-price-v1",
        "card": "22.5.C.2",
        "date": "2026-11-16",
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
    print(json.dumps({"unit": unit, "option": option}, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------- run


def run_seed(contract, seed, dataset, data, spend, root):
    model = b1.build_v4_reference(contract, seed)
    start = torch.load(ROOT / contract["start_bundles"][str(seed)], weights_only=False)
    codes = b1.codes_for(contract, model, data)
    head = c1.make_decode_head(model.c.d)
    c1.set_trainable_c1(model, contract)
    for p in head.parameters():
        p.requires_grad_(True)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
        contract["feedback_optimizer"],
    )
    margin = float(contract["plan"]["selectivity_margin"])
    a_baseline = b1.a_rows(model, data, codes)
    birth = c1.evaluate(model, head, dataset, codes, contract)
    birth_diag = diagnostics(model, head, dataset, codes)
    train = dataset["decode"]["train"]
    pairs = dataset["pairs"]
    updates = int(contract["plan"]["updates_per_seed"])
    for step in range(updates):
        spend()
        item = train[step % len(train)]
        pair = pairs[step % len(pairs)]
        train_step(model, head, optimizer, item, pair, codes[item["example"]["id"]], margin)
    final = c1.evaluate(model, head, dataset, codes, contract)
    final_diag = diagnostics(model, head, dataset, codes)
    a_final = b1.a_rows(model, data, codes)
    purity = c1.purity_check_c1(model, start, contract)
    save_path = root / f"seed-{seed}" / "bundle-post-training.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    record_state = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n in set(contract["record_parameters"])
    }
    bundle_sha = a5.p4_2.save_checkpoint(
        save_path,
        {
            "record_channel": record_state,
            "decode_head": {n: p.detach().clone() for n, p in head.state_dict().items()},
            "seed": seed,
            "revision": int(model.parameter_revision),
        },
    )
    return {
        "seed": seed,
        "updates": updates,
        "birth_decode_accuracy": birth["decode_fit_accuracy"],
        "birth_selectivity_accuracy": birth["selectivity_fit_accuracy"],
        "birth_diagnostics": birth_diag,
        "final_diagnostics": final_diag,
        "decode_fit_accuracy": final["decode_fit_accuracy"],
        "decode_fit_digest": final["decode_fit_digest"],
        "selectivity_fit_accuracy": final["selectivity_fit_accuracy"],
        "selectivity_fit_digest": final["selectivity_fit_digest"],
        "a_baseline_digest": a5.rows_digest(a_baseline),
        "a_final_digest": a5.rows_digest(a_final),
        "answer_path_bit_identical": a5.rows_digest(a_final) == a5.rows_digest(a_baseline),
        "purity": purity,
        "bundle_sha256": bundle_sha,
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
    dataset = c1.build_dataset(data, contract)
    root = ROOT / contract["run_root"]
    root.mkdir(parents=True, exist_ok=True)

    def spend() -> None:
        if time.process_time() - cpu_start > float(contract["plan"]["cap_cpu"]):
            raise BudgetStop("c2_budget_stop")
        if time.perf_counter() - wall_start > float(contract["plan"]["cap_wall"]):
            raise BudgetStop("c2_wall_stop")

    seeds = [int(value) for value in contract["plan"]["seeds"]]
    results = []
    for seed in seeds:
        spend()
        path = root / f"seed-{seed}-result.json"
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
        else:
            started = time.process_time()
            entry = run_seed(contract, seed, dataset, data, spend, root)
            entry["arm_cpu"] = round(time.process_time() - started, 6)
            recovery = recover_rounds(root / f"seed-{seed}" / "bundle-post-training.pt")
            entry["recovery"] = {
                "decode_digest_equal": recovery["decode_digest"] == entry["decode_fit_digest"],
                "selectivity_digest_equal": recovery["selectivity_digest"]
                == entry["selectivity_fit_digest"],
            }
            path.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        results.append(entry)

    criteria = contract["criteria"]
    decode_gate = float(criteria["decode_gate"])
    selectivity_gate = float(criteria["selectivity_gate"])
    decode_ok = all(e["decode_fit_accuracy"] >= decode_gate for e in results)
    selectivity_ok = all(e["selectivity_fit_accuracy"] >= selectivity_gate for e in results)
    purity_ok = all(e["purity"]["pure"] for e in results)
    answer_ok = all(e["answer_path_bit_identical"] for e in results)
    recovery_ok = all(
        e["recovery"]["decode_digest_equal"] and e["recovery"]["selectivity_digest_equal"]
        for e in results
    )
    if not (purity_ok and answer_ok and recovery_ok):
        verdict = "engineering_failure"
    elif decode_ok and selectivity_ok:
        verdict = "retrieval_representation_earned"
    elif not decode_ok and not selectivity_ok:
        verdict = "both_failed"
    elif not decode_ok:
        verdict = "content_decode_failed"
    else:
        verdict = "selectivity_failed"
    payload = {
        "schema": "taiji-u2-c2-position-run-v1",
        "card": "22.5.C.2",
        "date": "2026-11-16",
        "verdict": verdict,
        "seeds": seeds,
        "decode_fit_accuracies": [e["decode_fit_accuracy"] for e in results],
        "selectivity_fit_accuracies": [e["selectivity_fit_accuracy"] for e in results],
        "final_diagnostics": [e["final_diagnostics"] for e in results],
        "birth_diagnostics": [e["birth_diagnostics"] for e in results],
        "purity_ok": purity_ok,
        "answer_ok": answer_ok,
        "recovery_ok": recovery_ok,
        "precommitment": "if representation not earned, global closeout of the byte-world joint-gate effort (MAINLINE section 5)",
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
                "decode": payload["decode_fit_accuracies"],
                "selectivity": payload["selectivity_fit_accuracies"],
                "diag": payload["final_diagnostics"],
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
    dataset = c1.build_dataset(data, contract)
    model = b1.build_v4_reference(contract, seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in bundle["record_channel"]:
                p.copy_(bundle["record_channel"][n])
    model.eval()
    head = c1.make_decode_head(model.c.d)
    head.load_state_dict(bundle["decode_head"])
    head.eval()
    codes = b1.codes_for(contract, model, data)
    final = c1.evaluate(model, head, dataset, codes, contract)
    print(
        json.dumps(
            {
                "decode_digest": final["decode_fit_digest"],
                "selectivity_digest": final["selectivity_fit_digest"],
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
    dataset = c1.build_dataset(data, contract)
    checks = []
    for seed in [int(value) for value in contract["plan"]["seeds"]]:
        entry = json.loads((root / f"seed-{seed}-result.json").read_text(encoding="utf-8"))
        bundle = torch.load(root / f"seed-{seed}" / "bundle-post-training.pt", weights_only=False)
        model = b1.build_v4_reference(contract, seed)
        start = torch.load(ROOT / contract["start_bundles"][str(seed)], weights_only=False)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in bundle["record_channel"]:
                    p.copy_(bundle["record_channel"][n])
        model.eval()
        head = c1.make_decode_head(model.c.d)
        head.load_state_dict(bundle["decode_head"])
        head.eval()
        codes = b1.codes_for(contract, model, data)
        final = c1.evaluate(model, head, dataset, codes, contract)
        purity = c1.purity_check_c1(model, start, contract)
        checks.append({"check": f"purity_seed_{seed}", "ok": purity["pure"]})
        checks.append(
            {
                "check": f"decode_digest_seed_{seed}",
                "ok": final["decode_fit_digest"] == entry["decode_fit_digest"],
            }
        )
        checks.append(
            {
                "check": f"selectivity_digest_seed_{seed}",
                "ok": final["selectivity_fit_digest"] == entry["selectivity_fit_digest"],
            }
        )
        recovery = recover_rounds(root / f"seed-{seed}" / "bundle-post-training.pt")
        checks.append(
            {
                "check": f"recovery_seed_{seed}",
                "ok": recovery["decode_digest"] == entry["decode_fit_digest"]
                and recovery["selectivity_digest"] == entry["selectivity_fit_digest"],
            }
        )
    decode = [
        json.loads((root / f"seed-{s}-result.json").read_text(encoding="utf-8"))[
            "decode_fit_accuracy"
        ]
        for s in contract["plan"]["seeds"]
    ]
    checks.append({"check": "decode_match_report", "ok": decode == report["decode_fit_accuracies"]})
    payload = {
        "schema": "taiji-u2-c2-position-verify-v1",
        "card": "22.5.C.2",
        "date": "2026-11-16",
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks),
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
