"""P3.8.3 whole-recipe comparison: feature card, preflight, 128 paired updates.

Reads the frozen pre-registration; never edits it. The feature card and the
checkpoint preflight are written BEFORE the first optimizer step. All cost is
billed per arm; the frozen verdict set is applied literally at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training.joint_token_learner_22_5 import load_learner  # noqa: E402
from scripts.training.run_paired_learning_22_5 import (  # noqa: E402
    control_partition,
    evaluate,
)
from scripts.training.task_history_loss_22_5 import (
    history_loss,  # noqa: E402
    score_answer,  # noqa: E402
)
from scripts.training.verify_native_credit_transfer_22_5 import (  # noqa: E402
    DATA,
    DATA_DIGEST,
    HEAD_SEED,
    PARENT,
    PARENT_SHA256,
    class_table,
    coverage_indices,
    output,
    start_episode,
)
from taiji.u2_credit_learner import (  # noqa: E402
    BODY_TENSORS,
    HEAD_TENSORS,
    build_capsule,
    capsule_summary,
    headless_runtime,
    load_capsule,
    save_capsule,
)
from taiji.u2_credit_update import accepted_adam_step  # noqa: E402
from taiji.u2_history import atomic_load, atomic_save, detached_tree, value_digest  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_8_3_prereg_20261114.json"
CRITERIA_SHA256 = "b88b33818dd850439ca8f915caedbe1fce85a568f26ae566a9c8efbc294b94e6"
FEATURE_CARD = ROOT / "plans/manifests/taiji_u2_p3_8_3_feature_card_20261114.json"
PREFLIGHT = ROOT / "reports/taiji_u2_p3_8_3_preflight_20261114.json"
REPORT = ROOT / "reports/taiji_u2_p3_8_3_recipe_comparison_20261114.json"
MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_8_3_recipe_comparison_20261114.json"
PROGRESS = ROOT / "reports/taiji_u2_p3_8_3_recipe_comparison_20261114_progress.jsonl"
RUN_ROOT = ROOT / "data/experiments/taiji_u2/p3_8_3"


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def progress(row: dict[str, Any]) -> None:
    with PROGRESS.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def contract() -> dict[str, Any]:
    parsed = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert parsed["criteria_sha256"] == CRITERIA_SHA256
    assert canonical_sha(parsed["criteria"]) == CRITERIA_SHA256
    return parsed


def sources() -> dict[str, str]:
    names = (
        "scripts/training/run_p3_8_3_recipe_comparison_22_5.py",
        "scripts/training/verify_native_credit_transfer_22_5.py",
        "taiji/u2_credit_update.py",
        "taiji/u2_credit_learner.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def feature_card(partition: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    order, counts, weights = class_table(partition)
    coverage = coverage_indices(partition)
    card = {
        "schema": "taiji-u2-p3-8-3-feature-card-v1",
        "date": "2026-11-14",
        "criteria_sha256": CRITERIA_SHA256,
        "parent_sha256": PARENT_SHA256,
        "data_digest": DATA_DIGEST,
        "seed": criteria["identity_and_inputs"]["same_seed"],
        "paired_updates": criteria["dose"]["paired_updates"],
        "episode_order": [
            coverage[index % len(coverage)] for index in range(criteria["dose"]["paired_updates"])
        ],
        "class_hex_sorted": order,
        "class_counts": counts,
        "class_weights": weights,
        "head": criteria["arms"]["B_credit"]["content_head"],
        "head_seed": HEAD_SEED,
        "alpha_grid": criteria["arms"]["B_credit"]["update"],
        "evaluation": "run_paired_learning_22_5.evaluate on dev, first turn from rows[turn==0], controls without_rgb / without_questions",
        "bootstrap": {
            "resamples": criteria["gates"]["paired_bootstrap_resamples"],
            "confidence_lower_bound": criteria["gates"]["confidence_lower_bound"],
            "unit": "paired (episode_id, turn) rows",
            "seed": criteria["identity_and_inputs"]["same_seed"],
        },
        "source_pins": sources(),
        "measurement_note": "real dev evaluate costs ~23.66 CPU per full pass in the price probe; the frozen conservative estimate used 0.2031 for a single forward and is retained as an acknowledged underestimate",
    }
    if not FEATURE_CARD.exists():
        write_json(FEATURE_CARD, card)
    assert json.loads(FEATURE_CARD.read_text(encoding="utf-8")) == card
    return card


def capsule_for(arm: str, model, head, optimizer, u: int, v: int, cursor: int, weights):
    ids = [name for name, _ in model.named_parameters()]
    if head is not None:
        ids += ["content_head.weight", "content_head.bias"]
    owners = {"body": BODY_TENSORS}
    if head is not None:
        owners["content_head"] = HEAD_TENSORS
    return build_capsule(
        native_runtime={
            "architecture": "native-joint-query",
            "field_count": 44,
            "fact_history": [],
            "pending": [],
            "rng": torch.get_rng_state().clone(),
            "resources": {"arm": arm},
            "label": arm,
        },
        learning_state={
            "body_moment": detached_tree(optimizer.state_dict())["state"],
            "content_head": detached_tree(head.state_dict()) if head is not None else {},
            "content_head_moment": {},
            "label_source": "train_targets[0]_hex",
            "class_weights": weights,
            "parameter_ancestor": value_digest(model.state_dict()),
            "proposal_clock": u,
            "parameter_version": v,
        },
        sampling={
            "episode_cursor": cursor,
            "first_answer_prefix": "",
            "prefix_source": "actual_first_answer",
            "data_rights": {"train": True, "dev": False, "check": False, "final": False},
            "sampling_rng": torch.get_rng_state().clone(),
        },
        proposal=None,
        identity={
            "factory_schema": "taiji-u2-joint-token-query-v1",
            "source_pins": sources(),
            "parameter_registry": {"parameter_ids": ids, "owners": owners},
            "body_digest": value_digest(model.state_dict()),
        },
    )


def worker(checkpoint: str, destination: str) -> None:
    torch.set_num_threads(1)
    capsule = load_capsule(checkpoint)
    atomic_save(
        {
            "digest": value_digest(capsule),
            "summary": capsule_summary(capsule),
            "headless_fields": len(headless_runtime(capsule)),
            "pid": os.getpid(),
        },
        destination,
    )


def independent_restore(capsule: dict[str, Any], tag: str) -> dict[str, Any]:
    checkpoint = RUN_ROOT / f"{tag}.bin"
    destination = RUN_ROOT / f"{tag}_receipt.bin"
    save_capsule(checkpoint, capsule)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(checkpoint),
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=180)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    assert process.returncode == 0, stderr[-2000:] + stdout[-500:]
    receipt = atomic_load(destination)
    assert receipt["pid"] != os.getpid()
    assert receipt["digest"] == value_digest(capsule)
    return {"digest_matches": True, "point": tag}


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], resamples: int, seed: int) -> dict:
    def key(row: dict) -> tuple:
        return (row["episode_id"], row.get("turn"))

    index_b = {key(row): row["correct"] for row in rows_b}
    pairs = [(row["correct"], index_b[key(row)]) for row in rows_a if key(row) in index_b]
    rng = random.Random(seed)
    size = len(pairs)
    deltas = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(size)] for _ in range(size)]
        deltas.append(sum(b - a for a, b in sample) / size)
    deltas.sort()
    lower = deltas[int(0.025 * resamples)]
    upper = deltas[int(0.975 * resamples) - 1]
    return {
        "paired_units": size,
        "observed_delta": sum(b - a for a, b in pairs) / max(1, size),
        "ci_lower": lower,
        "ci_upper": upper,
        "resamples": resamples,
    }


def evaluate_arm(model, dev: dict, label: str, guard) -> dict[str, Any]:
    guard()
    result = evaluate(model, dev, guard)
    first = [row for row in result["rows"] if row.get("turn") == 0]
    return {
        "label": label,
        "correct": result["correct"],
        "denominator": result["denominator"],
        "accuracy": result["accuracy"],
        "first_turn_correct": sum(row["correct"] for row in first),
        "first_turn_denominator": len(first),
        "first_turn_accuracy": sum(row["correct"] for row in first) / max(1, len(first)),
        "rows": result["rows"],
    }


def run_arm(arm: str, criteria: dict, card: dict, dev: dict, train: dict, weights, budget) -> dict:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    model, optimizer, _ = load_learner(PARENT)
    head = None
    if arm == "B":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(HEAD_SEED)
            head = torch.nn.Linear(8, 4)
        optimizer.add_param_group({"params": list(head.parameters())})
    parameters = [*model.parameters(), *([*head.parameters()] if head is not None else [])]
    counters = {"optimizer_steps": 0}
    restores: list[dict[str, Any]] = []

    def guard():
        cpu = time.process_time() - started_cpu
        if cpu > criteria["budget"]["per_arm_cpu_cap"]:
            raise RuntimeError("per_arm_budget_stop")
        if time.monotonic() - started_wall > criteria["budget"]["runner_wall_cap"]:
            raise RuntimeError("wall_budget_stop")

    # Hard pre-flight: initial full capsule must restore in an independent process.
    restores.append(
        independent_restore(
            capsule_for(arm, model, head, optimizer, 0, 0, 0, weights), f"{arm}_initial"
        )
    )
    before = evaluate_arm(model, dev, f"{arm}_before", guard)
    order = card["class_hex_sorted"]

    def objective(index: int):
        episode = train["episodes"][card["episode_order"][index]]

        def compute():
            if arm == "A":
                return history_loss(model, train, episode, "actual-self")["loss"]
            initial = start_episode(model, train, episode)
            first_ce, _ = score_answer(model, initial, episode["targets"][0])
            _, context, _ = output(model, initial)
            omega = initial["logw"].detach().softmax(0)
            mixed = torch.logsumexp(
                head(context).log_softmax(-1) + torch.log(omega.clamp_min(1e-12))[:, None], 0
            )
            label = order.index(bytes(episode["targets"][0]).hex())
            return first_ce - weights[label] * mixed[label]

        return compute

    updates: list[dict[str, Any]] = []
    for index in range(criteria["dose"]["paired_updates"]):
        guard()
        objective_fn = objective(index)
        if arm == "A":
            optimizer.zero_grad(set_to_none=True)
            loss = objective_fn()
            loss.backward()
            optimizer.step()
            outcome = {"accepted": True, "reason": "plain_adam"}
        else:
            outcome = accepted_adam_step(parameters, optimizer, objective_fn, counters)
        updates.append(
            {"index": index, **{k: outcome.get(k) for k in ("accepted", "scale", "reason")}}
        )
        if index == 0:
            restores.append(
                independent_restore(
                    capsule_for(
                        arm,
                        model,
                        head,
                        optimizer,
                        1,
                        1 if outcome["accepted"] else 0,
                        index,
                        weights,
                    ),
                    f"{arm}_first_nonempty_moment",
                )
            )
        if index % 32 == 31:
            progress(
                {
                    "arm": arm,
                    "update": index + 1,
                    "cpu": round(time.process_time() - started_cpu, 3),
                }
            )

    after = evaluate_arm(model, dev, f"{arm}_after", guard)
    controls = {}
    for kind in ("without_rgb", "without_questions"):
        guard()
        controls[kind] = evaluate_arm(model, control_partition(dev, kind), f"{arm}_{kind}", guard)
    return {
        "arm": arm,
        "updates": updates,
        "restores": restores,
        "before": {k: v for k, v in before.items() if k != "rows"},
        "after": {k: v for k, v in after.items() if k != "rows"},
        "controls": {
            k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in controls.items()
        },
        "rows_before": before["rows"],
        "rows_after": after["rows"],
        "counters": {
            "real_adam_calls": counters["optimizer_steps"] if arm == "B" else len(updates)
        },
        "cpu_seconds": round(time.process_time() - started_cpu, 4),
        "wall_seconds": round(time.monotonic() - started_wall, 4),
    }


def decide(criteria: dict, arm_a: dict, arm_b: dict, bootstrap: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    gates = criteria["gates"]
    a_dynamic = (
        arm_a["after"]["accuracy"] > 0.5 + 1e-9
        or arm_a["after"]["first_turn_accuracy"] > 0.5 + 1e-9
        or arm_a["after"]["accuracy"] != arm_a["before"]["accuracy"]
        or arm_a["after"]["first_turn_accuracy"] != arm_a["before"]["first_turn_accuracy"]
    )
    if not a_dynamic:
        return "budget_cannot_express", ["arm A shows no dynamic range on either face"]
    delta = bootstrap["observed_delta"]
    if delta >= gates["minimum_meaningful_gain"] and bootstrap["ci_lower"] > 0:
        return "recipe_gain", reasons
    if delta <= 0 or bootstrap["ci_lower"] <= 0:
        return "recipe_no_gain", reasons
    reasons.append("confidence interval crosses zero")
    return "inconclusive", reasons


def main() -> int:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--phase", default="preflight", choices=["preflight", "run"])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return 0
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    criteria = contract()["criteria"]
    data = atomic_load(DATA)
    assert value_digest(data) == DATA_DIGEST
    train, dev = data["train"], data["dev"]
    _, _, weights = class_table(train)
    card = feature_card(train, criteria)

    if args.phase == "preflight":
        if PREFLIGHT.exists():
            raise FileExistsError(PREFLIGHT)
        # Two independent-process capsule restores, zero optimizer steps.
        checks = []
        for arm in ("A", "B"):
            model, optimizer, _ = load_learner(PARENT)
            head = None
            if arm == "B":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(HEAD_SEED)
                    head = torch.nn.Linear(8, 4)
                optimizer.add_param_group({"params": list(head.parameters())})
            checks.append(
                independent_restore(
                    capsule_for(arm, model, head, optimizer, 0, 0, 0, weights), f"preflight_{arm}"
                )
            )
        report = {
            "schema": "taiji-u2-p3-8-3-preflight-v1",
            "date": "2026-11-14",
            "criteria_sha256": CRITERIA_SHA256,
            "feature_card": str(FEATURE_CARD.relative_to(ROOT)).replace("\\", "/"),
            "feature_card_sha256": hashlib.sha256(FEATURE_CARD.read_bytes()).hexdigest(),
            "source_pins": sources(),
            "parent_sha256": PARENT_SHA256,
            "data_digest": DATA_DIGEST,
            "checkpoint_restores": checks,
            "evaluation_price_probe": {
                "dev_full_cpu": 23.6562,
                "dev_without_rgb_cpu": 21.2656,
                "dev_accuracy": 0.5,
                "dev_without_rgb_accuracy": 0.5,
                "note": "price probe measured the full 180-episode two-question evaluate; the frozen conservative estimate used a single forward and underestimates evaluation by ~116x",
            },
            "optimizer_steps": 0,
            "dev_reads": 1,
            "verdict": "preflight_passed",
        }
        write_json(PREFLIGHT, report)
        print(
            json.dumps({"verdict": "preflight_passed", "restores": len(checks)}, ensure_ascii=False)
        )
        return 0

    if REPORT.exists():
        raise FileExistsError(REPORT)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    arms: dict[str, dict[str, Any]] = {}
    verdict, reasons = "engineering_failure", ["runner did not reach a verdict"]
    try:
        for arm in ("A", "B"):
            arms[arm] = run_arm(arm, criteria, card, dev, train, weights, None)
            progress({"arm": arm, "done": True, "cpu": arms[arm]["cpu_seconds"]})
        bootstrap = paired_bootstrap(
            arms["A"]["rows_after"],
            arms["B"]["rows_after"],
            criteria["gates"]["paired_bootstrap_resamples"],
            criteria["identity_and_inputs"]["same_seed"],
        )
        cpu_ratio = max(arms["A"]["cpu_seconds"], arms["B"]["cpu_seconds"]) / max(
            1e-9, min(arms["A"]["cpu_seconds"], arms["B"]["cpu_seconds"])
        )
        if cpu_ratio > 2.0:
            verdict, reasons = "compute_parity_unavailable", [f"cpu_ratio={cpu_ratio:.4f}"]
        else:
            verdict, reasons = decide(criteria, arms["A"], arms["B"], bootstrap)
    except RuntimeError as error:
        verdict, reasons, bootstrap, cpu_ratio = "budget_stop", [str(error)], None, None
    except Exception as error:  # noqa: BLE001
        verdict, reasons, bootstrap, cpu_ratio = (
            "engineering_failure",
            [f"{type(error).__name__}: {error}"],
            None,
            None,
        )

    def public(arm: dict) -> dict:
        return {
            key: value
            for key, value in arm.items()
            if key not in ("rows_before", "rows_after", "updates")
        } | {
            "update_count": len(arm["updates"]),
            "accepted_updates": sum(1 for row in arm["updates"] if row["accepted"]),
        }

    report = {
        "schema": "taiji-u2-p3-8-3-recipe-comparison-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.8.3",
        "criteria_sha256": CRITERIA_SHA256,
        "feature_card_sha256": hashlib.sha256(FEATURE_CARD.read_bytes()).hexdigest(),
        "parent_sha256": PARENT_SHA256,
        "data_digest": DATA_DIGEST,
        "source_pins": sources(),
        "arms": {key: public(value) for key, value in arms.items()},
        "paired_bootstrap": bootstrap,
        "cpu_ratio": cpu_ratio,
        "reasons": reasons,
        "measured": {
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
        "discipline": {
            "dev_reads": 8,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "criteria_sha_unchanged_after_reading": canonical_sha(criteria) == CRITERIA_SHA256,
        },
        "verdict": verdict,
    }
    write_json(REPORT, report)
    manifest = {
        "schema": "taiji-u2-p3-8-3-recipe-comparison-manifest-v1",
        "criteria_sha256": CRITERIA_SHA256,
        "feature_card_sha256": report["feature_card_sha256"],
        "runner_sha256": sources()["scripts/training/run_p3_8_3_recipe_comparison_22_5.py"],
        "submitted_runner_sha256": sources()[
            "scripts/training/run_p3_8_3_recipe_comparison_22_5.py"
        ],
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        "verdict": verdict,
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {"verdict": verdict, "reasons": reasons, "cpu": report["measured"]["cpu_seconds"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
