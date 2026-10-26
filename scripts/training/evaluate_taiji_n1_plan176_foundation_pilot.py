"""One fixed F0/F1 sealed read after the three registered pilot endpoints."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from scripts.training.audit_taiji_n1_plan176_learning_paths import (
    CORPUS,
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
from scripts.training.run_taiji_n1_plan176_foundation_pilot import (
    ARMS,
    CHECKPOINTS,
    SELECTION,
    SPEC,
    report_path,
)
from scripts.training.run_taiji_r2_daily_dialogue_full_turn_pretraining import _json_selected_fields
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3
from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4

BASE = ROOT / "reports/taiji_n1_plan176_foundation_evaluation_20261026"
ATTEMPT = BASE.with_suffix(".attempt.json")


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
    related["response"] = color + "。"
    obj, old = row["metadata"]["distractor_pair"]
    color = alternative_color([obj, old])
    clause = f"{obj}是{old}的。"
    if row["user_input"].count(clause) != 1:
        raise ValueError("unrelated fact must have a unique source clause")
    irrelevant["user_input"] = irrelevant["user_input"].replace(clause, f"{obj}是{color}的。", 1)
    return related, irrelevant


def main() -> None:
    if BASE.exists() or ATTEMPT.exists():
        raise FileExistsError("sealed attempt already exists; preserve its outputs")
    torch.set_num_threads(1)
    cpu_start, wall_start = time.process_time(), time.monotonic()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    trainers, reports = {}, {}
    for arm in ARMS:
        report = json.loads(report_path(arm).read_text(encoding="utf-8"))
        if (
            report["status"] != "complete"
            or report["completed_updates"] != 2560
            or report["target_byte_exposures"] != 190080
            or report["sealed_text_read"] != 0
            or report["final_fresh_restore"]["status"] != "passed"
            or report["sources_sha256"]["spec"] != digest(SPEC)
            or report["sources_sha256"]["selection"] != digest(SELECTION)
            or report["sources_sha256"]["candidate"]
            != digest(ROOT / f"taiji/conditional_language_v{report['version']}.py")
        ):
            raise ValueError("pilot endpoint, exposure, separation or source gate failed")
        checkpoint = ROOT / report["final_checkpoint"]
        if digest(checkpoint) != report["final_checkpoint_sha256"]:
            raise ValueError("fixed endpoint checkpoint SHA differs")
        cls = (
            ConditionalLanguageTrainerV4 if report["version"] == 4 else ConditionalLanguageTrainerV3
        )
        trainer, _, cursor = cls.from_checkpoint(torch.load(checkpoint, weights_only=False))
        if cursor["next_batch"] != 2560 or cursor["order_sha256"] != report["order_sha256"]:
            raise ValueError("fixed endpoint cursor differs")
        trainers[arm], reports[arm] = trainer, report
    sealed = ROOT / corpus["output_directory"] / "sealed.jsonl"
    if digest(sealed) != corpus["split_sha256"]["sealed"]:
        raise ValueError("sealed source SHA differs")
    BASE.mkdir()
    with ATTEMPT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            {
                "status": "sealed_consumption_started_no_retry",
                "spec_sha256": digest(SPEC),
                "selection_sha256": digest(SELECTION),
                "evaluator_sha256": digest(Path(__file__)),
                "run_report_sha256": {a: digest(report_path(a)) for a in ARMS},
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    rows = []
    for line in sealed.read_text(encoding="utf-8").splitlines():
        if _json_selected_fields(line, {"curriculum_level"})["curriculum_level"] in {"F0", "F1"}:
            rows.append(json.loads(line))
    if {r["episode_id"] for r in rows} != set(selection["sealed_F0_F1_episode_ids"]):
        raise ValueError("frozen selected identities differ")
    records = []
    with (BASE / "outputs.jsonl").open("x", encoding="utf-8", newline="\n") as log:

        def emit(arm, model, row, condition, cut=False):
            if time.process_time() - cpu_start > 990:
                raise TimeoutError("fixed 1000 CPU-second evaluation guard reached")
            episode = as_language_episode(row)
            with torch.no_grad():
                state = prompt_state(model, episode.prompt_bytes)
                if cut:
                    state = replace(state, workspace=state.workspace[:0], evidence_refs=())
                raw = model.generate(state, max_new_bytes=96).generated
            item = {
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
            records.append(item)
            log.write(json.dumps(item) + "\n")
            log.flush()
            os.fsync(log.fileno())

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
        train_path = ROOT / corpus["output_directory"] / "train.jsonl"
        train_rows = {
            r["episode_id"]: r
            for r in (json.loads(x) for x in train_path.read_text(encoding="utf-8").splitlines())
        }
        for arm, trainer in trainers.items():
            cls = (
                ConditionalLanguageTrainerV4
                if reports[arm]["version"] == 4
                else ConditionalLanguageTrainerV3
            )
            earlier, _, _ = cls.from_checkpoint(
                torch.load(CHECKPOINTS / arm / "batch_00640.pt", weights_only=False)
            )
            for identity in selection["F0_retention_episode_ids"]:
                emit(arm, earlier.model, train_rows[identity], "F0_train_at640")
                emit(arm, trainer.model, train_rows[identity], "F0_train_final")
        old_pairs, _ = load_train_episodes()
        old_by_id = {e.episode_id: e for _, e in old_pairs}
        for arm, trainer in trainers.items():
            for identity in selection["N175_train_retention_episode_ids"]:
                episode = old_by_id[identity]
                with torch.no_grad():
                    raw = trainer.model.generate(
                        prompt_state(trainer.model, episode.prompt_bytes), max_new_bytes=128
                    ).generated
                hit, target = 0, episode.target_bytes[:128]
                for left, right in zip(raw, target, strict=False):
                    if left != right:
                        break
                    hit += 1
                item = {
                    "arm": arm,
                    "episode_id": identity,
                    "condition": "N175_train_retention",
                    "free_prefix_match": hit / len(target),
                    "generated_hex": raw.hex(),
                }
                records.append(item)
                log.write(json.dumps(item) + "\n")
                log.flush()
                os.fsync(log.fileno())
    if sealed_count != 1216 or len(records) != 1360:
        raise ValueError("fixed output count differs")
    result = {
        "schema": "taiji-n176-foundation-evaluation-v1",
        "status": "fixed_generation_complete",
        "sealed_outputs": sealed_count,
        "train_retention_outputs": len(records) - sealed_count,
        "attempt_sha256": digest(ATTEMPT),
        "F2_sealed_bodies_parsed": 0,
        "training_updates": 0,
        "n175_key_opened": False,
        "process_cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.monotonic() - wall_start,
        "can_promote": False,
    }
    with (BASE / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
