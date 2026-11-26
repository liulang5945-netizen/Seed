"""A.4: train the in-body column read together with the byte path.

The card answers one question: if the subject owns the read - request as query,
its own column tokens as keys and values - can it read the scene before it
answers, and can the byte path then say the answer in natural Chinese bytes?

Design notes that are decisions rather than details:

* The read is injected on the *input* side of the joint readout, so the decision
  point carries it. A.3 measured that the input carries no scene read, and a
  read added to the output would have left that measurement untouched.
* The read enters through a zero-initialised projection, so the first step is
  bit-identical to the body without the read.
* The four class probabilities are injected as well, so the byte loss earns the
  class instead of leaving the head outside the gradient path.
* The degraded first pass for the correction protocol is the trained subject with
  the read switched off (its projection zeroed). The pre-registered A.1b variant
  would have measured an external frozen head, that is, the pipeline, not the
  in-body read this card is about.

Gates are frozen in the contract before any reading:
  R  prefix-0 read class accuracy, margin over the shuffle null, painted-column
     colour probe - prefix 0 only
  B  full-response exact-byte success on all 32 fit rounds
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

import torch
from torch.nn import functional

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_p4_1_22_5 as p4_1
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_column_read import (
    ColumnRead,
    inject_read,
    patch_source,
    read_delta,
    read_for_example,
)
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_a4_column_read_20261116.json"
RUN_REPORT = ROOT / "reports/taiji_u2_a4_column_read_run_20261116.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_a4_column_read_price_20261116.json"
LABELS = ("左边", "中间", "右边", "未知")


class BudgetStop(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    if sha_of(Path(__file__)) != contract["source_pins"]["runner_sha256"]:
        raise ValueError("runner_source_drift")
    if sha_of(ROOT / "taiji/u2_column_read.py") != contract["source_pins"]["read_module_sha256"]:
        raise ValueError("read_module_source_drift")
    if mode == "run":
        if contract["status"] != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("training_authority_absent")
        if not contract.get("cpu_cap"):
            raise ValueError("cpu_cap_absent")


def build_subject(contract: dict, seed: int):
    candidate = contract["candidate"]
    bundle = ROOT / candidate["encoder_bundle"]
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    torch.manual_seed(int(seed))
    model = make_model("native", int(candidate["seed"]))
    model.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != candidate["parameters"]:
        raise ValueError("candidate_parameter_count_mismatch")
    if len(list(model.named_parameters())) != candidate["tensors"]:
        raise ValueError("candidate_tensor_count_mismatch")
    torch.manual_seed(int(seed) + 7)
    read = ColumnRead(int(candidate["width"]))
    return model, read


def set_trainable(model, read, contract: dict) -> list[str]:
    declared = set(contract["emission_parameters"])
    found = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in declared)
        if name in declared:
            found.append(name)
    if sorted(found) != sorted(declared):
        raise ValueError("trainable_set_mismatch")
    scalars = sum(p.numel() for p in model.parameters() if p.requires_grad)
    scalars += sum(p.numel() for p in read.parameters())
    if scalars != int(contract["trainable_scalar_count"]):
        raise ValueError("trainable_scalar_mismatch")
    return sorted(found)


def optimizer_for(model, read, spec: dict):
    parameters = [p for p in model.parameters() if p.requires_grad]
    parameters += list(read.parameters())
    return torch.optim.Adam(
        parameters,
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )


def cache_items(model, examples: list[dict]) -> list[dict]:
    """Cache what the frozen body passed to the emission operator, plus examples."""
    zeros = {example["id"]: torch.zeros(int(3 * model.c.d)) for example in examples}
    positions = a2.cache_positions(model, examples, zeros)
    for item in positions:
        item["example"] = next(e for e in examples if e["id"] == item["example"])
    return positions


def injection_of(model, read, example: dict) -> torch.Tensor:
    return read_for_example(model, read, example)["injection"]


def emission_loss(model, read, items: list[dict], read_weight: float = 0.0) -> torch.Tensor:
    """Byte emission loss with the read injected into the cached readout input.

    The optional read term is a cross entropy on the read's own four-way class
    against the public target word. The byte loss alone cannot teach the class:
    the class reaches the loss only through the residual projection, so its
    gradient is second order and died at exactly zero by step 50 in a trial.
    """
    cache: dict[str, torch.Tensor] = {}
    terms = []
    read_terms = []
    for item in items:
        key = item["example"]["id"]
        if key not in cache:
            features = read_for_example(model, read, item["example"])
            cache[key] = features["injection"]
            if read_weight:
                target = item["example"]["target"].decode()
                if target in LABELS:
                    read_terms.append(
                        functional.cross_entropy(
                            features["logits"][None], torch.tensor([LABELS.index(target)])
                        )
                    )
        source = patch_source(item["x"], read_delta(read, cache[key]))
        context = model.joint_readout(source)
        previous_value = model.previous_embedding(item["previous"])[None].expand(
            context.shape[0], -1
        )
        generator = model.byte_head(torch.cat([context, previous_value], -1)).log_softmax(-1)
        gate = model.copy_gate(
            torch.cat(
                [context, item["goal"][None].expand(context.shape[0], -1), previous_value], -1
            )
        ).sigmoid()
        lam, gamma = gate.unbind(-1)
        out = a2.subject_module.emission(
            item["cursor"],
            item["alpha"],
            item["beta"],
            lam,
            gamma,
            generator,
            item["ids"],
            item["valid"],
            item["mask"],
        )
        terms.append(-out["log_output"][item["symbol"]].mean())
    loss = torch.stack(terms).mean()
    if read_weight and read_terms:
        loss = loss + read_weight * torch.stack(read_terms).mean()
    return loss


def read_class(model, read, example: dict) -> int:
    with torch.no_grad():
        return int(read_for_example(model, read, example)["logits"].argmax())


def byte_rows(model, read, examples: list[dict], read_live: bool = True) -> list[dict]:
    """Greedy full-response rounds with the read injected at the readout input."""
    rows = []
    cap = int(model.c.max_output_bytes) + 1
    with torch.no_grad():
        for example in examples:
            state = p4_2.prepare(model, p4_2.column_public(example), example["id"])
            content: list[int] = []
            eos = False
            provider = (
                (lambda example=example: injection_of(model, read, example))
                if read_live
                else (lambda: None)
            )
            with inject_read(model, read, provider):
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
            rows.append(
                {
                    "id": example["id"],
                    "target": example["target"].decode(),
                    "output_hex": raw.hex(),
                    "eos": eos,
                    "valid_utf8": valid,
                    "correct": bool(eos and valid and raw == example["target"]),
                    "read_class": read_class(model, read, example),
                }
            )
    return rows


def rows_digest(rows: list[dict]) -> str:
    payload = json.dumps(
        [{k: v for k, v in row.items() if k != "read_class"} for row in rows],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def accuracy(rows: list[dict]) -> float:
    return sum(int(row["correct"]) for row in rows) / len(rows)


def read_accuracy(rows: list[dict]) -> float:
    hits = 0
    for row in rows:
        target = LABELS.index(row["target"]) if row["target"] in LABELS else -1
        hits += int(target == row["read_class"])
    return hits / len(rows)


def prefix_probes(model, read, data: dict) -> dict:
    """A.3's corrected instrument: prefix 0 only, with a shuffle null."""
    from scripts.training.review_a3_interface_20261116 import probe

    train = data["train"]
    fit = data["fit"]

    def capture_with_read() -> dict[str, list[torch.Tensor]]:
        positions = []
        for example in train + fit:
            features = injection_of(model, read, example)
            state = p4_2.prepare(model, p4_2.column_public(example), example["id"])
            captured: list[torch.Tensor] = []

            def provider(features=features):
                return features

            def capture_input(module, inputs, captured=captured):  # noqa: ANN001
                captured.append(inputs[0].detach().flatten())
                return None

            handle = model.joint_readout.register_forward_pre_hook(capture_input)
            try:
                with inject_read(model, read, provider):
                    model.output(state, model.language_goal(state), state["prediction_action"])
            finally:
                handle.remove()
            positions.append((example["id"], captured[0]))
        return {key: [value] for key, value in positions}

    grouped = capture_with_read()
    order = {example["id"]: index for index, example in enumerate(train + fit)}
    keys = [key for key, _ in sorted(grouped.items(), key=lambda pair: order[pair[0]])]
    x_train = torch.stack([grouped[key][0] for key in keys[: len(train)]])
    x_fit = torch.stack([grouped[key][0] for key in keys[len(train) :]])
    y_train = torch.tensor(
        [LABELS.index(example["target"].decode()) for example in train], dtype=torch.long
    )
    y_fit = torch.tensor(
        [LABELS.index(example["target"].decode()) for example in fit], dtype=torch.long
    )
    result = probe(x_train, y_train, x_fit, y_fit)
    return {
        "prefix": 0,
        "best_fit": result["best_fit"],
        "linear_fit": result["linear_fit"],
        "rbf_fit": result["rbf_fit"],
        "null_max": result["null_max"],
        "margin": result["best_fit"] - result["null_max"],
    }


def token_colour_probe(model, data: dict) -> dict:
    """A.3's colour instrument on the read's inputs: the column tokens.

    The painted-column number is the one the read depends on: if the tokens do
    not carry the queried colour, the attention cannot select the right column.
    """
    from scripts.training.review_a3_interface_20261116 import colour_probe

    train = data["train"]
    fit = data["fit"]

    def rows(examples: list[dict]):
        features, labels = [], []
        for example in examples:
            with torch.no_grad():
                features.append(p4_1.tokens_of(model, example).flatten())
            labels.append(p4_1.column_labels(example))
        return torch.stack(features), labels

    x_train, y_train = rows(train)
    x_fit, y_fit = rows(fit)
    return colour_probe(x_train, y_train, x_fit, y_fit)


def snapshot(model, read, contract: dict) -> dict:
    return {
        "read": read.state_dict(),
        "emission": {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if name in set(contract["emission_parameters"])
        },
    }


def decide(seed_rows: dict, contract: dict) -> str:
    """The ladder, with the premise separated from the outcomes.

    The painted-column colour number is a premise about the read's *inputs*: the
    encoder is frozen in this card, so that number cannot move during training.
    The read's own outcomes - its class accuracy and the prefix-0 margin - are
    what training can move. The thresholds are unchanged; the branch only says
    which of the two failed.
    """
    criteria = contract["criteria"]
    gate_r = criteria["gate_r_read_prefix0"]
    gate_b = criteria["gate_b_byte"]
    final = seed_rows["final"]
    probes = final["prefix_probes"]
    outcomes_ok = (
        final["read_accuracy_fit"] >= gate_r["fit_read_accuracy"]
        and probes["margin"] >= gate_r["margin_over_shuffle_null"]
        and probes["prefix"] == gate_r["prefix"]
    )
    premise_ok = probes["token_colour"]["painted"] >= gate_r["painted_column_colour_probe"]
    byte_ok = final["fit_accuracy"] >= gate_b["full_response_exact_success"]
    if outcomes_ok and byte_ok and premise_ok:
        return "read_in_body_confirmed"
    if outcomes_ok and byte_ok:
        return "premise_below_floor_read_outcomes_passed"
    if outcomes_ok:
        return "byte_gate_failed_with_read_passed"
    return "read_gate_failed"


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    seeds = [int(seed) for seed in contract["plan"]["seeds"]]
    doses = [int(dose) for dose in contract["dose_points"]]
    steps = int(contract["steps"])
    batch = int(contract["batch"])
    results = []
    for seed in seeds:
        model, read = build_subject(contract, seed)
        set_trainable(model, read, contract)
        optimizer = optimizer_for(model, read, contract["optimizer"])
        items = cache_items(model, data["train"] + data["fit"])
        cursor = 0
        dose_rows = []
        for update in range(1, steps + 1):
            if time.process_time() - cpu_start > contract["cpu_cap"]:
                raise BudgetStop("a4_budget_stop")
            batch_items = [items[(cursor + offset) % len(items)] for offset in range(batch)]
            cursor = (cursor + batch) % len(items)
            optimizer.zero_grad(set_to_none=True)
            loss = emission_loss(model, read, batch_items, float(contract["aux_read_weight"]))
            loss.backward()
            optimizer.step()
            if update not in doses:
                continue
            fit_rows = byte_rows(model, read, data["fit"])
            train_rows = byte_rows(model, read, data["train"])
            probes = prefix_probes(model, read, data)
            probes["token_colour"] = token_colour_probe(model, data)
            dose_rows.append(
                {
                    "update": update,
                    "fit_accuracy": accuracy(fit_rows),
                    "train_accuracy": accuracy(train_rows),
                    "read_accuracy_fit": read_accuracy(fit_rows),
                    "read_accuracy_train": read_accuracy(train_rows),
                    "fit_correct": sum(int(row["correct"]) for row in fit_rows),
                    "fit_denominator": len(fit_rows),
                    "distinct_outputs": len({row["output_hex"] for row in fit_rows}),
                    "prefix_probes": probes,
                    "fit_digest": rows_digest(fit_rows),
                }
            )
        final = dose_rows[-1]
        root = Path(contract["run_root"]) / f"seed-{seed}"
        root.mkdir(parents=True, exist_ok=True)
        # The bundle is taken here, at the final dose, so that it reproduces the
        # readings this report publishes. The guard checks exactly that; an
        # earlier version saved it after the feedback phase and the bundle no
        # longer matched the published numbers.
        torch.save(snapshot(model, read, contract), root / "bundle.pt")
        # degraded first pass: the trained subject with the read switched off
        degraded = byte_rows(model, read, data["fit"], read_live=False)
        errors = [row for row in degraded if not row["correct"]]
        control = byte_rows(model, read, data["fit"])
        learned = feedback_round(model, read, contract, data, errors, control)
        torch.save(snapshot(model, read, contract), root / "bundle-after-feedback.pt")
        results.append(
            {
                "seed": seed,
                "doses": dose_rows,
                "final": final,
                "degraded_first_pass": {
                    "fit_accuracy": accuracy(degraded),
                    "errors": len(errors),
                    "error_ids": [row["id"] for row in errors],
                },
                "correction": learned,
                "bundle": str((root / "bundle.pt").resolve().relative_to(ROOT)).replace("\\", "/"),
                "bundle_sha256": hashlib.sha256((root / "bundle.pt").read_bytes()).hexdigest(),
                "bundle_after_feedback": str(
                    (root / "bundle-after-feedback.pt").resolve().relative_to(ROOT)
                ).replace("\\", "/"),
                "bundle_after_feedback_sha256": hashlib.sha256(
                    (root / "bundle-after-feedback.pt").read_bytes()
                ).hexdigest(),
            }
        )
    primary = results[0]
    report = {
        "schema": "taiji-u2-a4-column-read-run-v1",
        "card": "22.5.A.4",
        "date": "2026-11-16",
        "verdict": decide(primary, contract),
        "seeds": results,
        "seed_direction": [
            row["final"]["fit_accuracy"]
            >= contract["criteria"]["gate_b_byte"]["full_response_exact_success"]
            for row in results
        ],
        "trainable_parameters": {
            "emission": int(contract["emission_scalar_count"]),
            "read": int(contract["candidate"]["read_scalars_expected"]),
            "total": int(contract["trainable_scalar_count"]),
        },
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "cost": {
            "total_cpu": round(time.process_time() - cpu_start, 6),
            "total_wall": round(time.perf_counter() - wall_start, 6),
            "cap_cpu": contract["cpu_cap"],
            "cap_wall": contract["wall_cap"],
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "scope": contract["criteria"]["scope_limit"],
    }
    RUN_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "final": primary["final"],
                "correction": primary["correction"],
                "cost": report["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def feedback_round(model, read, contract: dict, data: dict, errors, control_rows) -> dict:
    """Release the correct answers for the degraded errors, then re-ask once.

    The learned arm takes the feedback updates; the control arm takes none. Both
    re-ask all 32 fit rounds, so the gain is measured against a subject that saw
    the same released answers without learning from them.
    """
    spec = contract["feedback_optimizer"]
    learned_optimizer = optimizer_for(model, read, spec)
    by_id = {example["id"]: example for example in data["fit"]}
    for row in errors:
        example = by_id[row["id"]]
        state = p4_2.prepare(model, p4_2.column_public(example), example["id"])
        for _ in range(int(contract["feedback_updates"])):
            learned_optimizer.zero_grad(set_to_none=True)
            terms = []
            # The read is recomputed inside every update: reusing one graph
            # across eight backwards is the "backward through the graph a
            # second time" failure, not a modelling question.
            features = injection_of(model, read, example)
            with inject_read(model, read, lambda features=features: features):
                for symbol in [*example["target"], 256]:
                    probability, _, details = model.output(
                        state, model.language_goal(state), state["prediction_action"]
                    )
                    terms.append(-probability[symbol].log())
                    with torch.no_grad():
                        state = model.accept_byte(state, symbol, details)
            torch.stack(terms).mean().backward()
            learned_optimizer.step()
    learned_rows = byte_rows(model, read, data["fit"])
    control_accuracy = accuracy(control_rows)
    learned_accuracy = accuracy(learned_rows)
    fixed = sum(
        1
        for row in errors
        if next(item for item in learned_rows if item["id"] == row["id"])["correct"]
    )
    return {
        "first_pass_errors": len(errors),
        "corrected": fixed,
        "correction_rate": fixed / len(errors) if errors else 0.0,
        "learned_second_pass": learned_accuracy,
        "control_second_pass": control_accuracy,
        "feedback_gain": learned_accuracy - control_accuracy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run",), default="run")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    try:
        guards(contract, args.mode)
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
                        "schema": "taiji-u2-a4-column-read-run-v1",
                        "card": "22.5.A.4",
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
                        "schema": "taiji-u2-a4-column-read-run-v1",
                        "card": "22.5.A.4",
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
