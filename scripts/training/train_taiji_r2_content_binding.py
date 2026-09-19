"""Bounded training runner for the R2 content-binding package (contract §4/§6).

Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md

This runner is the ONLY statistical-training entry point of the package and it
is physically unable to see the sealed split: it loads the train fixture path
only, asserts every record carries ``split == "train"``, and exposes no
argument that could point at calibration or sealed data.  Calibration reads
happen exclusively inside ``evaluate_on_calibration`` (training-development
use, contract section 3); sealed-confirmation evaluation is a separately
authorised batch that lives outside this module.

Training is class-balanced at the GROUP level: a microbatch is 4 whole groups
(8 items), groups are never split, and a seeded sampler cycles the seven
classes so each receives equal exposure.

Statuses: ``completed`` (reached total updates with finite loss/grad),
``stopped_non_finite`` (contract stop rule), ``incomplete`` (resource cap hit;
never presents an earlier checkpoint as the prescribed endpoint).

Usage (frozen-training phase, requires separate budget approval):
    python scripts/training/train_taiji_r2_content_binding.py \
        --arm B --config cal_lr1 --seed 20260920 --total-updates 2000

``--mode smoke`` performs a wiring-only run of a few updates into an isolated
test directory; it is a test artifact, not a training run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_char_workspace import CharVocab  # noqa: E402
from taiji.sequence_content_workspace import (  # noqa: E402
    SEQUENCE_CONTENT_WORKSPACE_VERSION,
    SequenceContentConfig,
    SequenceContentTrainer,
    SequenceContentWorkspace,
)

TRAIN_FIXTURE = Path("tests/fixtures/r2_content_binding_v1_train.jsonl")
CALIBRATION_FIXTURE = Path("tests/fixtures/r2_content_binding_v1_calibration.jsonl")
OUTPUT_ROOT = Path("reports/r2_content_binding_v1")
CONTRACT = "plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md"

GROUP_CLASSES = (
    "fact_flip",
    "object_swap",
    "relation_flip",
    "negation_scope",
    "distractor_invariant",
    "missing_to_filled",
    "unknown_preserved",
)
FLIP_GATE_CLASSES = ("fact_flip", "object_swap", "relation_flip")
#: Contract section 4: calibration microbatch = 4 groups = 8 items.
BATCH_GROUPS = 4
#: Contract section 4 calibration evaluation points and caps.  v4 (contract
#: 20260920 §F1) extends to 8000 updates with dense margin observation.
CALIBRATION_POINTS = (500, 1000, 1500, 2000)
CALIBRATION_POINTS_V4 = (500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000)
CALIBRATION_MAX_UPDATES = 2000
CALIBRATION_V4_MAX_UPDATES = 8000
FORMAL_MAX_UPDATES = 2000
#: The two frozen candidate recipes (peak lr); everything else is shared.
RECIPES: dict[str, dict[str, float]] = {
    "cal_lr1": {"learning_rate": 0.001},
    "cal_lr3": {"learning_rate": 0.003},
}
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.1
HEALTH_EVERY = 100
SAVE_EVERY = 500
#: Selection rule (contract section 4).
SELECTION_COPY_MIN = 0.90
SELECTION_UNKNOWN_MIN = 0.90
WALL_CAP_SECONDS = 90 * 60  # per-run cap; the 12h package cap is bookkeeping


def load_train_fixture(path: Path | None = None) -> list[dict[str, Any]]:
    """Train-only loader: any foreign split row aborts the run."""

    source = PROJECT_ROOT / (TRAIN_FIXTURE if path is None else path)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    foreign = sorted({row["split"] for row in rows} - {"train"})
    if foreign:
        raise ValueError(
            f"train fixture carries non-train splits {foreign}; the training "
            "process must never see calibration or sealed data"
        )
    return rows


def load_calibration_fixture() -> list[dict[str, Any]]:
    source = PROJECT_ROOT / CALIBRATION_FIXTURE
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(row["split"] != "calibration" for row in rows):
        raise ValueError("calibration fixture carries a foreign split row")
    return rows


class GroupSampler:
    """Class-balanced, group-level sampler (contract section 4).

    A batch is ``batch_groups`` WHOLE groups drawn round-robin over a shuffled
    class cycle; each class's group queue reshuffles on epoch wrap, so groups
    repeat across epochs but are never split across batches.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        batch_groups: int = BATCH_GROUPS,
        seed: int = 0,
    ) -> None:
        self.batch_groups = int(batch_groups)
        self._rng = random.Random(f"content-binding-sampler:{seed}")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(record["group_id"], []).append(record)
        by_class: dict[str, list[list[dict[str, Any]]]] = {cls: [] for cls in GROUP_CLASSES}
        for members in grouped.values():
            members.sort(key=lambda r: r["member"])
            by_class[members[0]["group_class"]].append(members)
        self._all_groups = {cls: list(groups) for cls, groups in by_class.items()}
        self._queues = {cls: list(groups) for cls, groups in by_class.items()}
        for groups in self._queues.values():
            self._rng.shuffle(groups)
        self._cycle: list[str] = self._shuffled_classes()

    def _shuffled_classes(self) -> list[str]:
        classes = list(GROUP_CLASSES)
        self._rng.shuffle(classes)
        return classes

    def next_batch(self) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        while len(batch) < self.batch_groups * 2:  # two items per group
            if not self._cycle:
                self._cycle = self._shuffled_classes()
            cls = self._cycle.pop()
            queue = self._queues[cls]
            if not queue:
                queue[:] = list(self._all_groups[cls])
                self._rng.shuffle(queue)
            batch.extend(queue.pop())
        return batch


def train_pair_margin_diagnostic(
    workspace: Any,
    train_records: list[dict[str, Any]],
    *,
    per_class_limit: int = 32,
    gamma: float = 1.0,
) -> dict[str, Any]:
    """Contract v3 section E2 primary observable (train-only).

    Mean pair hinge per group class on TRAIN groups: hinge -> 0 means the
    model ranks its own member's answer above the crossed member's answer by
    at least gamma nats under the same prefix (margin satisfied); ~
    softplus(gamma) means the contrast was never realized.  Never touches
    calibration or sealed data.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in train_records:
        grouped.setdefault(record["group_id"], []).append(record)
    per_class: dict[str, list[float]] = {cls: [] for cls in GROUP_CLASSES}
    hinge_b: dict[str, list[float]] = {cls: [] for cls in GROUP_CLASSES}
    counts: dict[str, int] = {cls: 0 for cls in GROUP_CLASSES}
    with torch.no_grad():
        for members in grouped.values():
            cls = members[0]["group_class"]
            if counts[cls] >= per_class_limit:
                continue
            counts[cls] += 1
            members = sorted(members, key=lambda r: r["member"])
            a, b = members[0], members[1]
            s_xx, _ = workspace.sequence_loglik(a["question"], a["material"], a["response"])
            s_yy, _ = workspace.sequence_loglik(b["question"], b["material"], b["response"])
            s_yx, _ = workspace.sequence_loglik(a["question"], a["material"], b["response"])
            s_xy, _ = workspace.sequence_loglik(b["question"], b["material"], a["response"])
            per_class[cls].append(
                float(torch.nn.functional.softplus(s_yx - s_xx + gamma))
            )
            hinge_b[cls].append(
                float(torch.nn.functional.softplus(s_xy - s_yy + gamma))
            )
    return {
        cls: {
            "mean_hinge_a": sum(per_class[cls]) / len(per_class[cls]) if per_class[cls] else None,
            "mean_hinge_b": sum(hinge_b[cls]) / len(hinge_b[cls]) if hinge_b[cls] else None,
            "groups": counts[cls],
        }
        for cls in GROUP_CLASSES
    }


def evaluate_flip_scores(
    workspace: SequenceContentWorkspace, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Calibration-side scorer: per-class pair rates + copy/unknown rates.

    Only calibration data may reach this function (training-development use);
    the sealed split has no code path here.
    """

    scored: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            result = workspace.generate(record["question"], record["material"])
            scored.append(
                {
                    "id": record["id"],
                    "group_id": record["group_id"],
                    "group_class": record["group_class"],
                    "copyable": bool(record["copyable"]),
                    "gold": record["response"],
                    "prediction": result.text,
                    "correct": result.text == record["response"],
                    "range_error": result.range_error,
                    "stopped_on_boundary": result.stopped_on_boundary,
                }
            )
    by_class: dict[str, dict[str, Any]] = {}
    for cls in GROUP_CLASSES:
        members = [row for row in scored if row["group_class"] == cls]
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in members:
            groups.setdefault(row["group_id"], []).append(row)
        pair_hits = sum(
            1
            for members_of_group in groups.values()
            if len(members_of_group) == 2 and all(m["correct"] for m in members_of_group)
        )
        by_class[cls] = {
            "items": len(members),
            "item_exact": sum(1 for m in members if m["correct"]) / len(members) if members else None,
            "groups": len(groups),
            "pair_rate": pair_hits / len(groups) if groups else None,
        }
    copyable = [row for row in scored if row["copyable"]]
    unknown_items = [row for row in scored if row["gold"] == "未知"]
    flip_values = [by_class[cls]["pair_rate"] for cls in FLIP_GATE_CLASSES]
    finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in (
            workspace._parameters["char_embedding"],
            workspace._parameters["decoder"],
        )
    )
    return {
        "per_class": by_class,
        "flip_pairwise_macro": sum(flip_values) / len(flip_values) if flip_values else None,
        "copy_rate": (
            sum(1 for row in copyable if row["correct"]) / len(copyable)
            if copyable
            else None
        ),
        "unknown_rate": (
            sum(1 for row in unknown_items if row["correct"]) / len(unknown_items)
            if unknown_items
            else None
        ),
        "range_errors": sum(1 for row in scored if row["range_error"]),
        "finite_parameters": finite,
        "items": scored,
    }


def select_recipe_candidate(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Selection rule (contract section 4): among candidates meeting
    copy >= 0.90 and unknown >= 0.90 with finite values, the greatest
    three-flip-class pairwise macro wins; ties prefer the lower-lr recipe,
    then the earlier update."""

    eligible = [
        evaluation
        for evaluation in evaluations
        if evaluation["copy_rate"] is not None
        and evaluation["copy_rate"] >= SELECTION_COPY_MIN
        and evaluation["unknown_rate"] is not None
        and evaluation["unknown_rate"] >= SELECTION_UNKNOWN_MIN
        and evaluation["finite_parameters"]
        and evaluation["flip_pairwise_macro"] is not None
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda e: (-e["flip_pairwise_macro"], e["learning_rate"], e["update"])
    )
    return eligible[0]


def _display_path(path: Path) -> str:
    """Project-relative when possible; absolute for output roots outside the
    repository (tests write to the system temp dir, DEBT-I7 discipline)."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run(
    *,
    arm: str,
    config_name: str,
    seed: int,
    total_updates: int,
    output_root: Path = OUTPUT_ROOT,
    mode: str = "calibration",
    wall_cap_seconds: int = WALL_CAP_SECONDS,
    trainer_revision: str = "v1",
    question_hidden_width: int = 0,
    relation_hidden: int = 96,
    calibration_points: tuple[int, ...] = CALIBRATION_POINTS,
) -> dict[str, Any]:
    """One bounded training run.  Never called by this session's gate phase
    except through ``--mode smoke`` (test artifact)."""

    started = time.monotonic()
    records = load_train_fixture()
    digest = content_digest(records)
    vocab = CharVocab("".join(r["question"] + r["material"] + r["response"] for r in records))
    config = SequenceContentConfig(
        arm=arm,
        seed=seed,
        question_hidden_width=int(question_hidden_width) or 64,
        relation_hidden=int(relation_hidden),
    )
    torch.manual_seed(seed)
    workspace = SequenceContentWorkspace(vocab, config)
    recipe = RECIPES[config_name]
    trainer = SequenceContentTrainer(
        workspace,
        learning_rate=float(recipe["learning_rate"]),
        total_updates=int(total_updates),
        warmup_fraction=WARMUP_FRACTION,
        final_lr_fraction=FINAL_LR_FRACTION,
        code_revision=f"r2-content-binding-{mode}",
        data_digest=digest,
    )
    if trainer_revision == "v2":
        # contract v2 section D1: group-counterfactual contrastive auxiliary
        trainer.enable_pair_contrastive(weight=1.0, margin=1.0)
    elif trainer_revision != "v1":
        raise ValueError(f"unknown trainer revision: {trainer_revision}")
    run_dir = PROJECT_ROOT / output_root / arm / str(seed) / config_name
    run_dir.mkdir(parents=True, exist_ok=True)
    sampler = GroupSampler(records, seed=seed)

    health: list[dict[str, Any]] = []
    saved: list[str] = []
    exposure = {cls: 0 for cls in GROUP_CLASSES}  # groups drawn per class
    status = "completed"
    stop_reason: str | None = None
    latest_path = run_dir / "latest.pt"
    trainer.save(latest_path)
    saved.append(_display_path(latest_path))
    calibration_evaluations: list[dict[str, Any]] = []

    while trainer.global_step < int(total_updates):
        if time.monotonic() - started > wall_cap_seconds:
            status = "incomplete"
            stop_reason = "wall_cap"
            break
        batch = sampler.next_batch()
        for item in batch:
            exposure[item["group_class"]] += 0.5  # two items per group
        try:
            stats = trainer.train_step(batch)
        except FloatingPointError as error:
            status = "stopped_non_finite"
            stop_reason = str(error)
            break
        step = trainer.global_step
        if step % HEALTH_EVERY == 0 or step == total_updates:
            health.append({"update": step, **stats})
        if step % SAVE_EVERY == 0 or step == total_updates:
            path = run_dir / f"update_{step:06d}.pt"
            trainer.save(path)
            saved.append(_display_path(path))
            trainer.save(latest_path)
        if mode == "calibration" and step in calibration_points:
            evaluation = evaluate_flip_scores(workspace, load_calibration_fixture())
            if trainer.pair_contrastive_weight > 0.0:
                # contract v3 section E2 primary observable (train-only)
                evaluation["train_pair_margins"] = train_pair_margin_diagnostic(
                    workspace, records, gamma=float(trainer.pair_contrastive_margin)
                )
            evaluation.update(
                {
                    "update": step,
                    "learning_rate": float(recipe["learning_rate"]),
                    "checkpoint": _display_path(run_dir / f"update_{step:06d}.pt"),
                }
            )
            calibration_evaluations.append(evaluation)

    elapsed = time.monotonic() - started
    selected = select_recipe_candidate(calibration_evaluations) if calibration_evaluations else None
    report: OrderedDict[str, Any] = OrderedDict()
    report["format"] = "taiji-r2-content-binding-run-v1"
    report["version"] = 1
    report["contract"] = CONTRACT
    report["mode"] = mode
    report["arm"] = arm
    report["config"] = config_name
    report["recipe"] = recipe
    report["seed"] = int(seed)
    report["split_read"] = "train"
    report["calibration_read"] = mode == "calibration"
    report["sealed_read"] = False
    report["train_fixture"] = str(TRAIN_FIXTURE)
    report["data_digest"] = digest
    report["vocab_size"] = int(vocab.size)
    report["workspace_version"] = SEQUENCE_CONTENT_WORKSPACE_VERSION
    report["candidate_construction"] = "content-candidates-v1"
    report["selection_rule"] = "content-binding-selection-v1"
    report["trainer_revision"] = trainer.trainer_revision
    report["pair_contrastive_weight"] = float(trainer.pair_contrastive_weight)
    report["pair_contrastive_margin"] = float(trainer.pair_contrastive_margin)
    report["capacity"] = {
        "question_hidden_width": int(config.question_hidden_width),
        "relation_hidden": int(config.relation_hidden),
        "parameter_count": workspace.parameter_count(),
    }
    report["total_updates_planned"] = int(total_updates)
    report["updates_done"] = int(trainer.global_step)
    report["parameter_count"] = workspace.parameter_count()
    report["compute_profile"] = workspace.compute_profile()
    report["health"] = health
    report["group_exposure_per_class"] = exposure
    report["saved_checkpoints"] = saved
    report["calibration_evaluations"] = [
        {key: value for key, value in evaluation.items() if key != "items"}
        for evaluation in calibration_evaluations
    ]
    report["selected"] = None if selected is None else selected["update"]
    report["elapsed_seconds"] = elapsed
    report["wall_cap_seconds"] = wall_cap_seconds
    report["status"] = status
    report["stop_reason"] = stop_reason
    report["outcome"] = "passed" if status == "completed" else status
    report["can_promote"] = False
    out_path = run_dir / "run_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--config", choices=sorted(RECIPES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--total-updates",
        type=int,
        default=None,
        help="defaults: 2000 for calibration/formal per contract section 4",
    )
    parser.add_argument(
        "--mode",
        choices=("calibration", "formal", "smoke"),
        default="calibration",
        help="smoke = wiring-only test artifact into an isolated directory",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--wall-cap-seconds", type=int, default=WALL_CAP_SECONDS)
    parser.add_argument(
        "--trainer-revision",
        choices=("v1", "v2"),
        default="v1",
        help="v2 = pair-contrastive auxiliary (contract v2 section D1)",
    )
    parser.add_argument(
        "--question-hidden-width",
        type=int,
        default=64,
        help="question encoder width (contract v3: 128)",
    )
    parser.add_argument(
        "--relation-hidden",
        type=int,
        default=96,
        help="relation/content MLP width (contract v3: 128)",
    )
    parser.add_argument(
        "--calibration-points",
        default="500,1000,1500,2000",
        help="comma-separated calibration evaluation updates (v4: 500,1000,2000,...,8000)",
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        total = args.total_updates or 3
    elif args.total_updates is not None:
        total = int(args.total_updates)
        cap = CALIBRATION_MAX_UPDATES if args.mode == "calibration" else FORMAL_MAX_UPDATES
        if args.calibration_points != "500,1000,1500,2000":
            cap = 8000  # v4 contract: dense-point sweep carries an 8000-update cap
        if total > cap:
            raise SystemExit(f"--total-updates {total} exceeds the {args.mode} cap {cap}")
    else:
        total = CALIBRATION_MAX_UPDATES if args.mode == "calibration" else FORMAL_MAX_UPDATES

    if args.mode == "smoke":
        output_root = args.output_root / "_smoke"
    else:
        output_root = args.output_root

    report = run(
        arm=args.arm,
        config_name=args.config,
        seed=args.seed,
        total_updates=total,
        output_root=output_root,
        mode=args.mode,
        wall_cap_seconds=args.wall_cap_seconds,
        trainer_revision=args.trainer_revision,
        question_hidden_width=args.question_hidden_width,
        relation_hidden=args.relation_hidden,
        calibration_points=tuple(
            int(item) for item in args.calibration_points.split(",")
        ),
    )
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "arm": report["arm"],
                "config": report["config"],
                "updates_done": report["updates_done"],
                "status": report["status"],
                "final_loss": report["health"][-1]["loss"] if report["health"] else None,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
