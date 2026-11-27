"""15.1 rev2 stage 0c: same-form pricing for the four-arm binding-copy design.

Measures the declared costs of the four arms (R0P0, R1P0, R0P1, R1P1) on the
v2 train/dev split: batch-8 training updates at curriculum positions 1/101/201
with teacher-forcing and self-prefix branches, encoding, free generation,
NLL evaluation, intervention and independent restore.  Extrapolates the
15.1.1 two-seed x four-arm x 300-update run against the frozen cap.

Does not train for adoption, does not claim task capability, does not read the
sealed confirm split.  All engineering parameter copies and bundles are deleted
after measurement.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

# 单线程计费：与 14.5 定价脚本对齐，避免多线程放大 CPU 时间导致预算误判。
torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    BindingCopyConfig,
    answer_step,
    build_binding_candidate,
    generate_bytes,
    initial_binding_state,
    load_bundle,
    make_optimizer,
    observe_goal,
    prepare,
    save_bundle,
    score_example,
)
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    CONTENT_REVISION,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
)
from taiji.record_relation_core import answer_step as base_answer_step
from taiji.record_relation_core import generate_bytes as base_generate_bytes
from taiji.record_relation_core import load_bundle as base_load_bundle
from taiji.record_relation_core import make_optimizer as base_make_optimizer
from taiji.record_relation_core import observe_goal as base_observe_goal
from taiji.record_relation_core import prepare as base_prepare
from taiji.record_relation_core import save_bundle as base_save_bundle
from taiji.record_relation_core import score_example as base_score_example

PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_MANIFEST = "plans/manifests/taiji_core_record_relation_14_6_20261102.json"
GRAPH_REPORT = "reports/taiji_binding_copy_15_1_graph_20261102.json"
GRAPH_MANIFEST = "plans/manifests/taiji_binding_copy_15_1_graph_20261102.json"
DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"
REPORT = "reports/taiji_binding_copy_15_1_price_20261102.json"
MANIFEST = "plans/manifests/taiji_binding_copy_15_1_price_20261102.json"
OUTPUT_DIR = Path("output/taiji_core_binding_copy_15_1_price")

PRICE_CPU_SECONDS = 600.0
PRICE_WALL_SECONDS = 1200.0
PRICE_ADAM_CALLS = 12
PRICE_PRODUCT_BYTES = 1 * 1024**3
BATCH = 8
UPDATES_PER_ARM = 3
CURRICULUM_POSITIONS = (1, 101, 201)
WORKER_TIMEOUT = 240

# 15.1.1 cap: per owner instruction 18000 CPU / 36000 wall.
COMPARE_CPU_SECONDS = 18000
COMPARE_WALL_SECONDS = 36000
COMPARE_ADAM_CALLS = 2400  # 2 seeds x 4 arms x 300
COMPARE_PRODUCT_BYTES = 4 * 1024**3
COMPARE_SEEDS = 2
COMPARE_ARMS = 4
COMPARE_UPDATES_PER_ARM = 300
COMPARE_DEV_EXAMPLES = 240
COMPARE_EVAL_ENDPOINTS = 2  # step0 and step300
OVERHEAD_FRACTION = 0.10
STARTUP_WALL_SECONDS = 60.0

# Self-prefix probabilities by curriculum position (for P1 arms only).
SELF_PREFIX_PROB = {1: 0.0, 101: 0.5, 201: 0.75}

ARMS = ("R0P0", "R1P0", "R0P1", "R1P1")
ARM_READER = {"R0P0": "R0", "R1P0": "R1", "R0P1": "R0", "R1P1": "R1"}
ARM_PREFIX = {"R0P0": "TF", "R1P0": "TF", "R0P1": "self", "R1P1": "self"}

SOURCE_FILES = (
    "taiji/binding_copy.py",
    "taiji/binding_copy_data.py",
    "taiji/binding_copy_scoring.py",
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "scripts/training/verify_taiji_binding_copy_15_1_price.py",
)

STARTED_WALL = time.perf_counter()
STARTED_CPU = time.process_time()


class PriceStop(RuntimeError):
    """The declared price-stage budget was exhausted."""


class PriceBudget:
    def __init__(self) -> None:
        self.cpu_seconds = 0.0
        self.wall_seconds = 0.0
        self.adam_calls = 0
        self.product_bytes = 0

    def add(self, cpu: float, wall: float) -> None:
        self.cpu_seconds += cpu
        self.wall_seconds += wall
        self.ensure()

    def add_product(self, info: dict) -> None:
        self.product_bytes += info.get("bytes", 0)
        self.ensure()

    def ensure(self) -> None:
        if self.cpu_seconds > PRICE_CPU_SECONDS:
            raise PriceStop(f"cpu budget exceeded: {self.cpu_seconds:.2f} > {PRICE_CPU_SECONDS}")
        if self.wall_seconds > PRICE_WALL_SECONDS:
            raise PriceStop(f"wall budget exceeded: {self.wall_seconds:.2f} > {PRICE_WALL_SECONDS}")
        if self.adam_calls > PRICE_ADAM_CALLS:
            raise PriceStop(f"adam budget exceeded: {self.adam_calls} > {PRICE_ADAM_CALLS}")
        if self.product_bytes > PRICE_PRODUCT_BYTES:
            raise PriceStop(
                f"product budget exceeded: {self.product_bytes} > {PRICE_PRODUCT_BYTES}"
            )


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def build_base_from_parent(
    payload: dict, *, dtype: torch.dtype = torch.float64
) -> RecordRelationCandidate:
    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    candidate = RecordRelationCandidate(subject, config, ARM_FULL, list(payload["growth_gates"]))
    named = dict(candidate.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    candidate.to(dtype)
    candidate.configure_trainable()
    return candidate


def example_to_fixture(example: dict) -> dict:
    records = tuple(
        RawSource(source_id=r["source_id"], raw=r["raw"].encode("utf-8"))
        for r in example["records"]
    )
    target_bytes = tuple(example["action"].encode("utf-8"))
    return {
        "index": example["index"],
        "goal": example["goal"],
        "records": records,
        "target_bytes": target_bytes + (EOS,),
        "target_action": example["action"],
    }


def load_data() -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    payload = json.loads(Path(DATA_PAYLOAD).read_text(encoding="utf-8"))
    examples = payload["examples"]
    train = tuple(example_to_fixture(e) for e in examples if e["split"] == "train")
    dev = tuple(example_to_fixture(e) for e in examples if e["split"] == "dev")
    return train, dev


def self_prefix_bytes(candidate, fixture: dict, *, r1: bool, subject_id: str) -> tuple[int, ...]:
    """Greedy-generate the target-length prefix in no_grad mode."""
    gen_fn = generate_bytes if r1 else base_generate_bytes
    result = gen_fn(candidate, fixture["goal"].encode(), fixture["records"], subject_id=subject_id)
    raw = result["raw_bytes"]
    target_len = len(fixture["target_bytes"]) - 1  # exclude EOS
    # Truncate or pad with EOS to target length.
    prefix = tuple(raw[:target_len])
    if len(prefix) < target_len:
        prefix = prefix + (EOS,) * (target_len - len(prefix))
    return prefix


def batch_loss(
    candidate, fixtures: tuple[dict, ...], *, r1: bool, self_flags: tuple[bool, ...] | None
) -> tuple[torch.Tensor, float]:
    score_fn = score_example if r1 else base_score_example
    losses = []
    self_gen_cpu = 0.0
    for idx, fixture in enumerate(fixtures):
        use_self = bool(self_flags[idx]) if self_flags else False
        if use_self:
            start = time.process_time()
            with torch.no_grad():
                prefix = self_prefix_bytes(
                    candidate, fixture, r1=r1, subject_id=f"price-self-{fixture['index']}"
                )
            self_gen_cpu += time.process_time() - start
        else:
            prefix = fixture["target_bytes"][:-1]
        result = score_fn(candidate, fixture["goal"].encode(), fixture["records"], prefix)
        losses.append(result.nll)
    return sum(losses) / len(losses), self_gen_cpu


def measure_update(
    candidate,
    optimizer,
    fixtures: tuple[dict, ...],
    *,
    r1: bool,
    self_flags: tuple[bool, ...] | None,
) -> dict:
    start = time.process_time()
    loss, self_gen_cpu = batch_loss(candidate, fixtures, r1=r1, self_flags=self_flags)
    forward = time.process_time() - start
    start = time.process_time()
    optimizer.zero_grad()
    loss.backward()
    bptt = time.process_time() - start
    start = time.process_time()
    clip = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    step = time.process_time() - start
    return {
        "loss": float(loss.detach()),
        "forward_cpu": forward,
        "self_gen_cpu": self_gen_cpu,
        "bptt_cpu": bptt,
        "step_cpu": step,
        "total_cpu": forward + bptt + step,
        "clip_scale": clip,
    }


def measure_encoding(candidate, fixtures: tuple[dict, ...], *, r1: bool) -> float:
    prep_fn = prepare if r1 else base_prepare
    if r1:
        state = initial_binding_state(candidate, "price-encode")
    else:
        state = candidate.subject.initial_state("price-encode", revision=CONTENT_REVISION)
    start = time.process_time()
    for fixture in fixtures:
        prep_fn(candidate, state, fixture["goal"].encode(), fixture["records"])
    return (time.process_time() - start) / len(fixtures)


def measure_generation(
    candidate, fixtures: tuple[dict, ...], *, r1: bool
) -> tuple[float, list[dict]]:
    gen_fn = generate_bytes if r1 else base_generate_bytes
    results = []
    start = time.process_time()
    with torch.no_grad():
        for fixture in fixtures:
            results.append(
                gen_fn(
                    candidate,
                    fixture["goal"].encode(),
                    fixture["records"],
                    subject_id=f"price-gen-{fixture['index']}",
                )
            )
    elapsed = (time.process_time() - start) / len(fixtures)
    return elapsed, [
        {"index": f["index"], "terminated": r["terminated"], "raw_bytes": len(r["raw_bytes"])}
        for f, r in zip(fixtures, results, strict=True)
    ]


def measure_eval_score(candidate, fixtures: tuple[dict, ...], *, r1: bool) -> float:
    score_fn = score_example if r1 else base_score_example
    start = time.process_time()
    with torch.no_grad():
        for fixture in fixtures:
            score_fn(
                candidate,
                fixture["goal"].encode(),
                fixture["records"],
                fixture["target_bytes"][:-1],
                subject_id=f"price-eval-{fixture['index']}",
            )
    return (time.process_time() - start) / len(fixtures)


def measure_intervention(candidate, fixtures: tuple[dict, ...], *, r1: bool) -> float:
    prep_fn = prepare if r1 else base_prepare
    obs_fn = observe_goal if r1 else base_observe_goal
    step_fn = answer_step if r1 else base_answer_step
    start = time.process_time()
    with torch.no_grad():
        for fixture in fixtures:
            if r1:
                state = initial_binding_state(candidate, f"price-force-{fixture['index']}")
            else:
                state = candidate.subject.initial_state(
                    f"price-force-{fixture['index']}", revision=CONTENT_REVISION
                )
            prepared = prep_fn(candidate, state, fixture["goal"].encode(), fixture["records"])
            observation, gate_index = obs_fn(candidate, prepared)
            if r1:
                # gamma_override=0 forces generator-only path as a content-path intervention.
                step_fn(
                    candidate,
                    prepared,
                    observation,
                    EOS,
                    gate_index=gate_index,
                    gamma_override=0.0,
                )
            else:
                step_fn(
                    candidate,
                    prepared,
                    observation,
                    observation.coverage,
                    EOS,
                    gate_index=gate_index,
                )
    return (time.process_time() - start) / len(fixtures)


def measure_restore(
    candidate,
    optimizer,
    fixtures: tuple[dict, ...],
    *,
    r1: bool,
    output_dir: Path,
    tag: str,
    update_index: int,
    budget: PriceBudget,
) -> dict:
    save_fn = save_bundle if r1 else base_save_bundle
    load_fn = load_bundle if r1 else base_load_bundle
    if r1:
        state = initial_binding_state(candidate, f"price-restore-{tag}")
    else:
        state = candidate.subject.initial_state(f"price-restore-{tag}", revision=CONTENT_REVISION)
    bundle_path = output_dir / f"{tag}.pt"
    start = time.process_time()
    info = save_fn(
        bundle_path,
        profile=tag,
        candidate=candidate,
        state=state,
        optimizer=optimizer,
        update_index=update_index,
        cursor={"tag": tag},
        parent_identity={},
    )
    save_cpu = time.process_time() - start
    budget.add_product(info)
    start = time.process_time()
    loaded = load_fn(bundle_path, torch.float64)
    load_cpu = time.process_time() - start
    del loaded
    bundle_path.unlink(missing_ok=True)
    return {
        "save_cpu": save_cpu,
        "load_cpu": load_cpu,
        "elapsed_cpu": save_cpu + load_cpu,
        "bundle_mib": info.get("bytes", 0) / (1024 * 1024),
        "restore_ok": True,
    }


def run_arm(
    parent: dict,
    train: tuple[dict, ...],
    dev: tuple[dict, ...],
    *,
    arm: str,
    output_dir: Path,
    budget: PriceBudget,
    rng: torch.Generator,
) -> dict:
    reader = ARM_READER[arm]
    prefix_mode = ARM_PREFIX[arm]
    r1 = reader == "R1"
    if r1:
        base = build_base_from_parent(parent, dtype=torch.float64)
        candidate = build_binding_candidate(
            base, config=BindingCopyConfig(), profile=f"{arm}-f64", dtype=torch.float64
        )
        optimizer = make_optimizer(candidate)
    else:
        candidate = build_base_from_parent(parent, dtype=torch.float64)
        optimizer = base_make_optimizer(candidate)

    train_batch = train[:BATCH]
    dev_batch = dev[:BATCH]

    initial_restore = measure_restore(
        candidate,
        optimizer,
        dev_batch,
        r1=r1,
        output_dir=output_dir,
        tag=f"{arm}-initial",
        update_index=0,
        budget=budget,
    )
    encoding_cpu = measure_encoding(candidate, train_batch, r1=r1)

    update_rows = []
    for position in CURRICULUM_POSITIONS:
        if prefix_mode == "self":
            prob = SELF_PREFIX_PROB[position]
            self_flags = tuple(
                bool(torch.rand(1, generator=rng).item() < prob) for _ in range(BATCH)
            )
        else:
            self_flags = tuple(False for _ in range(BATCH))
        row = measure_update(candidate, optimizer, train_batch, r1=r1, self_flags=self_flags)
        row["position"] = position
        row["self_count"] = sum(self_flags)
        update_rows.append(row)
        budget.adam_calls += 1
        budget.ensure()

    generation_cpu, generation_rows = measure_generation(candidate, dev_batch, r1=r1)
    eval_cpu = measure_eval_score(candidate, dev_batch, r1=r1)
    intervention_cpu = measure_intervention(candidate, dev_batch, r1=r1)
    final_restore = measure_restore(
        candidate,
        optimizer,
        dev_batch,
        r1=r1,
        output_dir=output_dir,
        tag=f"{arm}-final",
        update_index=UPDATES_PER_ARM,
        budget=budget,
    )

    result = {
        "arm": arm,
        "reader": reader,
        "prefix_mode": prefix_mode,
        "parameter_count": candidate.parameter_count(),
        "trainable_count": int(sum(p.numel() for p in candidate.trainable_parameters())),
        "encoding_cpu": encoding_cpu,
        "updates": update_rows,
        "generation_cpu": generation_cpu,
        "generation": generation_rows,
        "eval_score_cpu": eval_cpu,
        "intervention_cpu": intervention_cpu,
        "initial_restore": initial_restore,
        "final_restore": final_restore,
        "adam_calls": UPDATES_PER_ARM,
    }
    del candidate
    gc.collect()
    return result


def frozen_prices(arms: list[dict]) -> dict:
    updates = [row for arm in arms for row in arm["updates"]]
    return {
        "train_update_batch8_cpu": median([row["total_cpu"] for row in updates]),
        "forward_batch8_cpu": median([row["forward_cpu"] for row in updates]),
        "self_gen_batch8_cpu": median([row.get("self_gen_cpu", 0.0) for row in updates]),
        "bptt_batch8_cpu": median([row["bptt_cpu"] for row in updates]),
        "adam_step_batch8_cpu": median([row["step_cpu"] for row in updates]),
        "encode_example_cpu": median([arm["encoding_cpu"] for arm in arms]),
        "self_generate_example_cpu": median([arm["generation_cpu"] for arm in arms]),
        "eval_score_example_cpu": median([arm["eval_score_cpu"] for arm in arms]),
        "intervention_example_cpu": median([arm["intervention_cpu"] for arm in arms]),
        "restore_cpu": median([arm["final_restore"]["elapsed_cpu"] for arm in arms]),
        "bundle_mib": median([arm["final_restore"]["bundle_mib"] for arm in arms]),
    }


def extrapolate(prices: dict) -> dict:
    train_updates = COMPARE_SEEDS * COMPARE_ARMS * COMPARE_UPDATES_PER_ARM
    eval_generations = COMPARE_SEEDS * COMPARE_ARMS * COMPARE_EVAL_ENDPOINTS * COMPARE_DEV_EXAMPLES
    eval_scores = eval_generations
    interventions = COMPARE_SEEDS * COMPARE_ARMS * COMPARE_EVAL_ENDPOINTS * COMPARE_DEV_EXAMPLES
    restores = COMPARE_SEEDS * COMPARE_ARMS
    training_cpu = train_updates * prices["train_update_batch8_cpu"]
    generation_cpu = eval_generations * prices["self_generate_example_cpu"]
    score_cpu = eval_scores * prices["eval_score_example_cpu"]
    intervention_cpu = interventions * prices["intervention_example_cpu"]
    restore_cpu = restores * prices["restore_cpu"]
    subtotal = training_cpu + generation_cpu + score_cpu + intervention_cpu + restore_cpu
    overhead = OVERHEAD_FRACTION * subtotal
    total_cpu = subtotal + overhead
    total_wall = total_cpu + STARTUP_WALL_SECONDS
    products_mib = restores * prices["bundle_mib"] + 100.0
    admitted = (
        total_cpu <= COMPARE_CPU_SECONDS
        and total_wall <= COMPARE_WALL_SECONDS
        and train_updates <= COMPARE_ADAM_CALLS
    )
    return {
        "formula": {
            "train_updates": train_updates,
            "train_batch": BATCH,
            "eval_generations": eval_generations,
            "eval_scores": eval_scores,
            "interventions": interventions,
            "restores": restores,
            "overhead_fraction": OVERHEAD_FRACTION,
            "startup_wall_seconds": STARTUP_WALL_SECONDS,
        },
        "components": {
            "training_cpu": training_cpu,
            "generation_cpu": generation_cpu,
            "eval_score_cpu": score_cpu,
            "intervention_cpu": intervention_cpu,
            "restore_cpu": restore_cpu,
            "subtotal_cpu": subtotal,
            "overhead_cpu": overhead,
            "total_cpu": total_cpu,
            "estimated_wall_seconds": total_wall,
            "estimated_products_mib": products_mib,
        },
        "cap": {
            "cpu_seconds": COMPARE_CPU_SECONDS,
            "wall_seconds": COMPARE_WALL_SECONDS,
            "adam_calls": COMPARE_ADAM_CALLS,
            "product_bytes": COMPARE_PRODUCT_BYTES,
        },
        "admitted": admitted,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="price", choices=("price",))
    args = parser.parse_args()
    if args.stage != "price":
        raise SystemExit("only the price stage is implemented")

    budget = PriceBudget()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(38031)

    try:
        parent = torch.load(ROOT / PARENT_BUNDLE, map_location="cpu", weights_only=False)
        train, dev = load_data()

        arm_results = []
        for arm in ARMS:
            result = run_arm(
                parent, train, dev, arm=arm, output_dir=OUTPUT_DIR, budget=budget, rng=rng
            )
            arm_results.append(result)

        prices = frozen_prices(arm_results)
        extrap = extrapolate(prices)

        status = "priced_within_compare_cap" if extrap["admitted"] else "budget_not_admitted"
        total_cpu = time.process_time() - STARTED_CPU
        total_wall = time.perf_counter() - STARTED_WALL

        source_hashes = {f: sha256_text(ROOT / f) for f in SOURCE_FILES}

        report = {
            "schema": "taiji-binding-copy-price-v1",
            "step": "15.1.0c",
            "date": "2026-11-02",
            "status": status,
            "arms": arm_results,
            "prices": prices,
            "extrapolation": extrap,
            "decision": {
                "admitted": extrap["admitted"],
                "reason": None if extrap["admitted"] else "price exceeds 15.1.1 cap",
            },
            "cost": {
                "cpu_seconds": total_cpu,
                "wall_seconds": total_wall,
                "adam_calls": budget.adam_calls,
                "product_bytes": budget.product_bytes,
            },
            "budget": {
                "cpu_seconds": PRICE_CPU_SECONDS,
                "wall_seconds": PRICE_WALL_SECONDS,
                "adam_calls": PRICE_ADAM_CALLS,
                "product_bytes": PRICE_PRODUCT_BYTES,
            },
            "source_hashes": source_hashes,
            "limits": "Pricing only. No training for adoption, no confirm read, no capability claim. Parent bundle is read-only.",
        }

        Path(REPORT).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        manifest = {
            "schema": "taiji-binding-copy-manifest-v1",
            "step": "15.1.0c",
            "status": status,
            "report": REPORT,
            "parent_bundle": PARENT_BUNDLE,
            "source_hashes": source_hashes,
            "admitted": extrap["admitted"],
            "frozen_prices": prices,
        }
        Path(MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        # Clean up output bundles.
        for p in OUTPUT_DIR.glob("*.pt"):
            p.unlink(missing_ok=True)

        print(f"0c pricing status: {status}")
        print(
            f"  cpu: {total_cpu:.2f}/{PRICE_CPU_SECONDS}, wall: {total_wall:.2f}/{PRICE_WALL_SECONDS}"
        )
        print(f"  adam: {budget.adam_calls}/{PRICE_ADAM_CALLS}")
        print(f"  extrapolated cpu: {extrap['components']['total_cpu']:.2f}/{COMPARE_CPU_SECONDS}")
        print(
            f"  extrapolated wall: {extrap['components']['estimated_wall_seconds']:.2f}/{COMPARE_WALL_SECONDS}"
        )
        return 0

    except PriceStop as exc:
        print(f"0c pricing budget stop: {exc}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
