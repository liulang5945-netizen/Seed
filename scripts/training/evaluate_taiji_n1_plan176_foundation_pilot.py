"""One fixed F0/F1 sealed read after the three registered pilot endpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from scripts.training.audit_taiji_n1_plan176_learning_paths import (
    CORPUS,
    PARENT,
    PARENT_REPORT,
    alternative_color,
    digest,
    normalized_answer,
    prompt_state,
)
from scripts.training.build_taiji_n1_plan176_foundation_corpus import (
    COLORS,
    OBJECTS,
    ROOT,
    as_language_episode,
    pair_split,
)
from scripts.training.calibrate_taiji_n1_plan175_v3_train_only import load_train_episodes
from scripts.training.migrate_taiji_n1_plan176_v3_to_v4 import OUTPUT as V4_PARENT
from scripts.training.run_taiji_n1_plan176_foundation_pilot import (
    ARMS,
    CHECKPOINTS,
    SELECTION,
    SPEC,
    report_path,
    schedule,
)
from scripts.training.run_taiji_r2_daily_dialogue_full_turn_pretraining import _json_selected_fields
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3
from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4
from taiji.internalization import content_digest

BASE = ROOT / "reports/taiji_n1_plan176_foundation_evaluation_20261026"
ATTEMPT = BASE.with_suffix(".attempt.json")
COSTS = BASE.with_suffix(".costs.jsonl")
N175_SELECTION = ROOT / "reports/taiji_n1_plan175_v3_train_rollout_selection_20261026.json"
CPU_LIMIT = 1000


@dataclass
class EvaluationBudget:
    """Charge preflight, all generations and failed preflights to the same limit."""

    cpu_start: float
    wall_start: float
    prior_cpu_seconds: float = 0.0

    def cpu_seconds(self) -> float:
        return time.process_time() - self.cpu_start

    def check(self) -> None:
        # Leave a small reserve for preserving the output and cost receipt.
        if self.prior_cpu_seconds + self.cpu_seconds() > CPU_LIMIT - 10:
            raise TimeoutError("fixed 1000 CPU-second evaluation guard reached")


def durable_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prior_evaluation_cpu() -> float:
    if not COSTS.exists():
        return 0.0
    costs = [json.loads(line)["process_cpu_seconds"] for line in COSTS.read_text().splitlines()]
    if any(not isinstance(value, (int, float)) or not 0 <= value < float("inf") for value in costs):
        raise ValueError("invalid durable evaluation cost ledger")
    return sum(costs)


def record_cost(payload: dict) -> None:
    with COSTS.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_ids(values: list[str], count: int, label: str) -> set[str]:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != count
    ):
        raise ValueError(f"fixed {label} identities differ")
    return set(values)


def checkpoint_body_digest(payload: dict) -> str:
    return content_digest(
        {key: value for key, value in payload.items() if key not in {"cursor", "checkpoint_digest"}}
    )


def preflight(budget: EvaluationBudget) -> dict:
    """Load every required train-side dependency; never open a sealed file."""
    budget.check()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    train = ROOT / corpus["output_directory"] / "train.jsonl"
    train_sha = digest(train)
    if (
        train_sha != corpus["split_sha256"]["train"]
        or train_sha != selection["source_train_sha256"]
        or corpus["split_sha256"]["sealed"] != selection["source_sealed_sha256"]
        or selection["registered_before_training"] is not True
    ):
        raise ValueError("frozen train corpus or registered selection source differs")
    all_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    if any(row["split"] != "train" for row in all_rows):
        raise ValueError("foundation train source contains a nontraining row")
    rows = sorted(
        (row for row in all_rows if row["curriculum_level"] in {"F0", "F1"}),
        key=lambda row: row["episode_id"],
    )
    train_ids = require_ids(selection["train_episode_ids"], 640, "training")
    if len(rows) != 640 or {row["episode_id"] for row in rows} != train_ids:
        raise ValueError("frozen training identities differ")
    train_rows = {row["episode_id"]: row for row in rows}
    f0_ids = require_ids(selection["F0_retention_episode_ids"], 16, "F0 retention")
    if not f0_ids <= {row["episode_id"] for row in rows if row["curriculum_level"] == "F0"}:
        raise ValueError("F0 retention probe left the fixed F0 training pool")
    require_ids(selection["sealed_F0_F1_episode_ids"], 256, "sealed F0/F1")
    old_ids = require_ids(selection["N175_train_retention_episode_ids"], 16, "N175 retention")
    old_selection = json.loads(N175_SELECTION.read_text(encoding="utf-8"))
    if old_ids != require_ids(old_selection["episode_ids"], 16, "original N175 retention"):
        raise ValueError("N175 retention probe differs from the original R16 baseline")
    episodes = [as_language_episode(row) for row in rows]
    orders, order_hashes, prefix_exposures = {}, {}, {}
    for arm in ARMS:
        batches = schedule(rows, arm)
        orders[arm] = batches
        order_hashes[arm] = hashlib.sha256(
            json.dumps(
                [[rows[index]["episode_id"] for index in batch] for batch in batches]
            ).encode()
        ).hexdigest()
        prefix_exposures[arm] = {
            count: sum(
                len(episodes[index].target_bytes) for batch in batches[:count] for index in batch
            )
            for count in (0, 640, 2560)
        }
        if len(batches) != 2560 or prefix_exposures[arm][2560] != 190080:
            raise ValueError("fixed schedule length or target exposure differs")
    if orders["curriculum_v3"] != orders["curriculum_v4"]:
        raise ValueError("curriculum arms must share the exact fixed order")
    budget.check()
    trainers, earlier, checkpoint_sources = {}, {}, {}
    parents = {}
    for version, path, receipt_path, sha_field in (
        (3, PARENT, PARENT_REPORT, "final_checkpoint_sha256"),
        (
            4,
            V4_PARENT,
            ROOT / "reports/taiji_n1_plan176_v4_parent_migration_20261026.json",
            "checkpoint_sha256",
        ),
    ):
        budget.check()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        parent_sha = digest(path)
        if parent_sha != receipt[sha_field]:
            raise ValueError("original uncalibrated parent source differs")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            content_digest(
                {key: value for key, value in payload.items() if key != "checkpoint_digest"}
            )
            != payload["checkpoint_digest"]
        ):
            raise ValueError("original parent checkpoint content digest differs")
        parents[version] = (path, parent_sha, payload)
        budget.check()
    for arm in ARMS:
        budget.check()
        version = 4 if arm == "curriculum_v4" else 3
        cls = ConditionalLanguageTrainerV4 if version == 4 else ConditionalLanguageTrainerV3
        report = json.loads(report_path(arm).read_text(encoding="utf-8"))
        source_paths = {
            "spec": SPEC,
            "selection": SELECTION,
            "runner": ROOT / "scripts/training/run_taiji_n1_plan176_foundation_pilot.py",
            "candidate": ROOT / f"taiji/conditional_language_v{version}.py",
            "restore_helper": ROOT / "scripts/training/calibrate_taiji_n1_plan176_foundation.py",
        }
        parent_path, parent_sha, parent_payload = parents[version]
        if (
            report["status"] != "complete"
            or report["arm"] != arm
            or report["version"] != version
            or report["planned_updates"] != 2560
            or report["completed_updates"] != 2560
            or report["planned_target_byte_exposures"] != 190080
            or report["target_byte_exposures"] != 190080
            or report["sealed_text_read"] != 0
            or report["n175_key_opened"] is not False
            or report["train_corpus_sha256"] != train_sha
            or report["order_sha256"] != order_hashes[arm]
            or (ROOT / report["parent_checkpoint"]).resolve() != parent_path.resolve()
            or report["parent_checkpoint_sha256"] != parent_sha
            or report["initial_steps"] != parent_payload["steps"]
            or report["initial_steps"] != 16520
            or report["initial_target_byte_exposures"] != parent_payload["target_byte_exposures"]
            or report["zero_fresh_restore"]["status"] != "passed"
            or report["zero_fresh_restore"]["steps"] != report["initial_steps"]
            or report["final_fresh_restore"]["status"] != "passed"
            or report["final_fresh_restore"]["steps"] != report["initial_steps"] + 2560
            or any(
                report["sources_sha256"][name] != digest(path)
                for name, path in source_paths.items()
            )
        ):
            raise ValueError(
                f"{arm}: pilot endpoint, lineage, order, exposure or source gate failed"
            )
        durable = {item["next_batch"]: item for item in report["durable_checkpoints"]}
        if len(durable) != len(report["durable_checkpoints"]):
            raise ValueError("duplicate durable checkpoint cursor")
        checkpoint_sources[arm] = {}
        for count in (0, 640, 2560):
            budget.check()
            path = CHECKPOINTS / arm / ("zero.pt" if count == 0 else f"batch_{count:05d}.pt")
            checkpoint_sha = digest(path)
            if count:
                item = durable[count]
                if (ROOT / item["path"]).resolve() != path.resolve() or item[
                    "sha256"
                ] != checkpoint_sha:
                    raise ValueError("fixed durable checkpoint path or SHA differs")
            if count == 2560 and (
                (ROOT / report["final_checkpoint"]).resolve() != path.resolve()
                or report["final_checkpoint_sha256"] != checkpoint_sha
            ):
                raise ValueError("fixed endpoint checkpoint path or SHA differs")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            trainer, _, cursor = cls.from_checkpoint(payload)
            if (
                cursor["next_batch"] != count
                or cursor["order_sha256"] != order_hashes[arm]
                or trainer.steps != report["initial_steps"] + count
                or trainer.target_byte_exposures
                != report["initial_target_byte_exposures"] + prefix_exposures[arm][count]
                or (count and (cursor["arm"] != arm or cursor["train_corpus_sha256"] != train_sha))
            ):
                raise ValueError("fixed checkpoint cursor, step counter or exposure differs")
            if not count and checkpoint_body_digest(payload) != checkpoint_body_digest(
                parent_payload
            ):
                raise ValueError("zero checkpoint differs from its original uncalibrated parent")
            checkpoint_sources[arm][str(count)] = checkpoint_sha
            if count == 640:
                earlier[arm] = trainer
            elif count == 2560:
                trainers[arm] = trainer
            budget.check()
    budget.check()
    old_pairs, old_sources = load_train_episodes()
    budget.check()
    old_by_id = {episode.episode_id: episode for _, episode in old_pairs}
    if len(old_by_id) != len(old_pairs) or not old_ids <= old_by_id.keys():
        raise ValueError("N175 retention identities are missing or duplicated in the train pool")
    if any(not old_by_id[identity].target_bytes for identity in old_ids):
        raise ValueError("N175 retention target is empty")
    return {
        "corpus": corpus,
        "selection": selection,
        "train_rows": train_rows,
        "trainers": trainers,
        "earlier": earlier,
        "old_by_id": {identity: old_by_id[identity] for identity in old_ids},
        "provenance": {
            "train_corpus_sha256": train_sha,
            "checkpoint_sha256": checkpoint_sources,
            "run_report_sha256": {arm: digest(report_path(arm)) for arm in ARMS},
            "N175_train_sources": old_sources,
            "N175_selection_sha256": digest(N175_SELECTION),
            "evaluation_helpers_sha256": {
                name: digest(ROOT / name)
                for name in (
                    "scripts/training/audit_taiji_n1_plan176_learning_paths.py",
                    "scripts/training/build_taiji_n1_plan176_foundation_corpus.py",
                    "scripts/training/calibrate_taiji_n1_plan175_v3_train_only.py",
                    "scripts/data_prep/audit_taiji_r2_daily_dialogue_material_inventory.py",
                    "taiji/conditional_language_v2.py",
                )
            },
        },
    }


def color_variants(row: dict) -> tuple[dict, dict]:
    related, irrelevant = copy.deepcopy(row), copy.deepcopy(row)
    obj, old = row["metadata"]["primary_pair"]
    color = next(
        c
        for i, c in enumerate(COLORS)
        if c != old and pair_split(OBJECTS.index(obj), i) == row["split"]
    )
    clause = f"{obj}是{old}的。"
    if row["user_input"].count(clause) != 1:
        raise ValueError("queried fact must have a unique source clause")
    related["user_input"] = related["user_input"].replace(clause, f"{obj}是{color}的。", 1)
    related["metadata"]["primary_pair"][1] = color
    related["metadata"]["answer_from_world"] = color
    related["response"] = color + "。"
    obj, old = row["metadata"]["distractor_pair"]
    color = alternative_color([obj, old])
    clause = f"{obj}是{old}的。"
    if row["user_input"].count(clause) != 1:
        raise ValueError("unrelated fact must have a unique source clause")
    irrelevant["user_input"] = irrelevant["user_input"].replace(clause, f"{obj}是{color}的。", 1)
    irrelevant["metadata"]["distractor_pair"][1] = color
    return related, irrelevant


def generate_fixed_outputs(
    prepared: dict, rows: list[dict], budget: EvaluationBudget, log, records: list
) -> None:
    """Write the fixed 1360 outputs, including train retention, under one budget."""
    selection, train_rows = prepared["selection"], prepared["train_rows"]
    trainers = prepared["trainers"]

    def save(item):
        log.write(json.dumps(item) + "\n")
        log.flush()
        os.fsync(log.fileno())
        records.append(item)
        budget.check()

    def emit(arm, model, row, condition, cut=False):
        budget.check()
        episode = as_language_episode(row)
        with torch.no_grad():
            state = prompt_state(model, episode.prompt_bytes)
            if cut:
                state = replace(state, workspace=state.workspace[:0], evidence_refs=())
            raw = model.generate(state, max_new_bytes=96).generated
        save(
            {
                "arm": arm,
                "episode_id": row["episode_id"],
                "level": row["curriculum_level"],
                "query_kind": row["metadata"]["query_kind"],
                "condition": condition,
                "correct": normalized_answer(raw) == normalized_answer(episode.target_bytes),
                "valid_utf8": normalized_answer(raw) is not None,
                "terminated": raw.endswith(b"<|end|>"),
                "generated_hex": raw.hex(),
            }
        )

    for arm, trainer in trainers.items():
        for row in rows:
            emit(arm, trainer.model, row, "matched")
            if row["metadata"]["query_kind"] == "primary_color":
                related, irrelevant = color_variants(row)
                emit(arm, trainer.model, related, "related_fact_flip")
                emit(arm, trainer.model, irrelevant, "irrelevant_fact_flip")
                if arm == "curriculum_v4":
                    emit(arm, trainer.model, row, "workspace_cut", cut=True)
    sealed_count = len(records)
    for arm, trainer in trainers.items():
        for identity in selection["F0_retention_episode_ids"]:
            emit(arm, prepared["earlier"][arm].model, train_rows[identity], "F0_train_at640")
            emit(arm, trainer.model, train_rows[identity], "F0_train_final")
    for arm, trainer in trainers.items():
        for identity in selection["N175_train_retention_episode_ids"]:
            budget.check()
            episode = prepared["old_by_id"][identity]
            with torch.no_grad():
                raw = trainer.model.generate(
                    prompt_state(trainer.model, episode.prompt_bytes), max_new_bytes=128
                ).generated
            hit, target = 0, episode.target_bytes[:128]
            for left, right in zip(raw, target, strict=False):
                if left != right:
                    break
                hit += 1
            save(
                {
                    "arm": arm,
                    "episode_id": identity,
                    "condition": "N175_train_retention",
                    "free_prefix_match": hit / len(target),
                    "generated_hex": raw.hex(),
                }
            )
    if sealed_count != 1216 or len(records) != 1360:
        raise ValueError("fixed output count differs")
    budget.check()


def selected_f0_f1_rows(lines, budget: EvaluationBudget) -> list[dict]:
    """Extract only selected levels; an F2 body never reaches the JSON decoder."""
    rows = []
    for line in lines:
        budget.check()
        if _json_selected_fields(line, {"curriculum_level"})["curriculum_level"] in {"F0", "F1"}:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    if BASE.exists() or ATTEMPT.exists():
        raise FileExistsError("sealed attempt already exists; preserve its outputs")
    torch.set_num_threads(1)
    budget = EvaluationBudget(time.process_time(), time.monotonic(), prior_evaluation_cpu())
    records = []
    attempt_registered, complete, failure = False, False, None
    try:
        prepared = preflight(budget)
        corpus, selection = prepared["corpus"], prepared["selection"]
        sealed = ROOT / corpus["output_directory"] / "sealed.jsonl"
        budget.check()
        if digest(sealed) != corpus["split_sha256"]["sealed"]:
            raise ValueError("sealed source SHA differs")
        budget.check()
        BASE.mkdir()
        with ATTEMPT.open("x", encoding="utf-8", newline="\n") as stream:
            # Even a failed or incomplete attempt record blocks all regeneration.
            attempt_registered = True
            json.dump(
                {
                    "status": "sealed_consumption_started_no_retry",
                    "spec_sha256": digest(SPEC),
                    "selection_sha256": digest(SELECTION),
                    "evaluator_sha256": digest(Path(__file__)),
                    "preflight": prepared["provenance"],
                    "prior_evaluation_cpu_seconds": budget.prior_cpu_seconds,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        with sealed.open("r", encoding="utf-8") as stream:
            rows = selected_f0_f1_rows(stream, budget)
        if len(rows) != 256 or {r["episode_id"] for r in rows} != set(
            selection["sealed_F0_F1_episode_ids"]
        ):
            raise ValueError("frozen selected identities differ")
        if (
            sum(row["curriculum_level"] == "F0" for row in rows) != 128
            or sum(row["curriculum_level"] == "F1" for row in rows) != 128
            or sum(row["metadata"]["query_kind"] == "primary_color" for row in rows) != 64
            or sum(row["metadata"]["query_kind"] == "distractor_color" for row in rows) != 64
        ):
            raise ValueError("fixed sealed curriculum or color-query counts differ")
        with (BASE / "outputs.jsonl").open("x", encoding="utf-8", newline="\n") as log:
            generate_fixed_outputs(prepared, rows, budget, log, records)
        budget.check()
        complete = True
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        sealed_count = sum(
            row["condition"] not in {"F0_train_at640", "F0_train_final", "N175_train_retention"}
            for row in records
        )
        cpu_seconds = budget.cpu_seconds()
        result = {
            "schema": "taiji-n176-foundation-evaluation-v1",
            "status": (
                "fixed_generation_complete"
                if complete
                else (
                    "sealed_consumption_failed_no_retry"
                    if attempt_registered
                    else "preflight_failed_no_sealed_consumption"
                )
            ),
            "sealed_outputs": sealed_count,
            "train_retention_outputs": len(records) - sealed_count,
            "attempt_sha256": digest(ATTEMPT) if attempt_registered else None,
            "F2_sealed_bodies_parsed": 0,
            "training_updates": 0,
            "n175_key_opened": False,
            "process_cpu_seconds": cpu_seconds,
            "prior_evaluation_cpu_seconds": budget.prior_cpu_seconds,
            "cumulative_evaluation_cpu_seconds": budget.prior_cpu_seconds + cpu_seconds,
            "wall_seconds": time.monotonic() - budget.wall_start,
            "failure": failure,
            "can_promote": False,
        }
        # The append-only cost journal also preserves failed preflight costs;
        # it does not register a sealed attempt or permit a second generation.
        record_cost(result)
        if attempt_registered:
            durable_json(BASE / "receipt.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
