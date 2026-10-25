"""After all three runs finish, generate the fixed family-isolated blind review set."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest
from scripts.training.calibrate_taiji_n1_plan175_v3_train_only import build_episode
from scripts.training.migrate_taiji_n1_plan175_v2_to_v3 import SEEDS
from scripts.training.run_taiji_n1_plan175_v3_scale import report_path
from scripts.training.run_taiji_r2_daily_dialogue_full_turn_pretraining import _json_selected_fields
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3

PARTITION = ROOT / "reports/taiji_n1_plan175_metadata_partition_20261026.json"
MIGRATION = ROOT / "reports/taiji_n1_plan175_v3_parent_migration_20261026.json"
CORPUS = ROOT / "reports/taiji_n1_plan175_v3_train_corpus_v2_20261026.json"
RAW_SOURCE = ROOT / "data/conversation/OASST2/2023-11-05_oasst2_ready.messages.jsonl.gz"
REPORTS = ROOT / "reports"
OUTPUT_DIR = REPORTS / "taiji_n1_plan175_v3_review_20261026"
HOLDOUT_SALT = "n175-root-holdout-v1-20261026"
MAX_INPUT_BYTES = 4096
MAX_OUTPUT_BYTES = 2048


def select_one_per_family(candidates: list[dict]) -> list[dict]:
    chosen, seen = [], set()
    for row in sorted(
        candidates,
        key=lambda item: hashlib.sha256(f"{HOLDOUT_SALT}:{item['prompt_id']}".encode()).hexdigest(),
    ):
        if row["family"] not in seen:
            seen.add(row["family"])
            chosen.append(row)
    return sorted(chosen, key=lambda item: item["family"])


def require_completed_runs() -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:
        path = report_path(seed)
        if not path.is_file():
            raise ValueError("N1-175 holdout remains sealed until all three runs complete")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("epochs_complete") != 40:
            raise ValueError(f"seed {seed} has not reached its fixed 40-epoch endpoint")
        checkpoint = ROOT / report["final_checkpoint"]
        if digest(checkpoint) != report["final_checkpoint_sha256"]:
            raise ValueError(f"seed {seed} final checkpoint SHA differs")
        runs.append({"seed": seed, "report": report, "report_sha256": digest(path)})
    return runs


def load_fixed_holdout() -> tuple[list[dict], list[dict], int, list[dict]]:
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    selected = select_one_per_family(partition["holdout_candidates"])
    if len(partition["holdout_families"]) != 64 or len(selected) != 64:
        raise ValueError("frozen holdout no longer has exactly one preselected prompt per family")
    rows = {}
    fields = {"message_id", "parent_id", "role", "lang", "deleted"}
    with gzip.open(RAW_SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = _json_selected_fields(line, fields)
            identity = row.get("message_id")
            if not identity or identity in rows:
                raise ValueError("frozen source has a missing or duplicate ID")
            rows[identity] = row
    chains, needed = {}, set()
    for candidate in selected:
        identity, chain, seen = candidate["target_id"], [], set()
        while identity is not None:
            if identity not in rows or identity in seen:
                raise ValueError("selected holdout has a broken/cyclic ancestry chain")
            seen.add(identity)
            chain.append(identity)
            identity = rows[identity].get("parent_id")
        chain.reverse()
        if chain[0] != candidate["family"]:
            raise ValueError("selected holdout family differs from source ancestry")
        chains[candidate["target_id"]] = chain
        needed.update(chain)
    with gzip.open(RAW_SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            identity = _json_selected_fields(line, {"message_id"}).get("message_id")
            if identity in needed:
                rows[identity]["text"] = _json_selected_fields(line, {"text"}).get("text")
    episodes, excluded = [], []
    for candidate in selected:
        chain = [rows[identity] for identity in chains[candidate["target_id"]]]
        try:
            for index, row in enumerate(chain):
                expected = {"user", "prompter"} if index % 2 == 0 else {"assistant"}
                if row.get("deleted") or row.get("lang") != "zh" or row.get("role") not in expected:
                    raise ValueError("invalid role/language/deleted history node")
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError("empty source text")
            episode = build_episode(candidate, chain)
            if (
                len(episode.prompt_bytes) > MAX_INPUT_BYTES
                or len(episode.target_bytes) > MAX_OUTPUT_BYTES
            ):
                raise ValueError("fixed prompt/target byte bound exceeded")
            episodes.append((candidate, episode))
        except (TypeError, ValueError) as exc:
            excluded.append(
                {
                    "candidate": candidate,
                    "reason": str(exc),
                }
            )
    return episodes, excluded, len(needed), selected


def generate_for_seed(seed: int, report: dict, episodes: list[tuple[dict, Any]]) -> list[dict]:
    checkpoint = ROOT / report["final_checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer, _, cursor = ConditionalLanguageTrainerV3.from_checkpoint(payload)
    if cursor.get("epoch") != 40 or trainer.steps != 5000 + 40 * 288:
        # Exact update count is additionally checked against the saved per-epoch ledger below.
        expected_steps = 5000 + sum(item["updates"] for item in report["epoch_results"])
        if cursor.get("epoch") != 40 or trainer.steps != expected_steps:
            raise ValueError(f"seed {seed} trainer cursor differs from its final report")
    wrong_indices = {index: (index + 1) % len(episodes) for index in range(len(episodes))}
    outputs = []
    for index, (candidate, episode) in enumerate(episodes):
        for condition, source_episode in (
            ("matched", episode),
            ("mismatched_prompt", episodes[wrong_indices[index]][1]),
        ):
            state = trainer.model.new_state(
                scope_id="n1-175-final",
                session_id=f"seed-{seed}",
                native_state_ref=f"n1-175:holdout:{candidate['family']}:{condition}",
            )
            state = trainer.model.step_task(
                source_episode.prompt_bytes,
                state,
                scope_id="n1-175-final",
                session_id=f"seed-{seed}",
                source_id="prompt",
                resource_snapshot={
                    "remaining_steps": 1,
                    "max_observation_bytes": MAX_INPUT_BYTES,
                },
            )
            generated = trainer.model.generate(state, max_new_bytes=MAX_OUTPUT_BYTES)
            opaque = hashlib.sha256(
                f"n175-review:{seed}:{candidate['family']}:{condition}".encode()
            ).hexdigest()[:16]
            outputs.append(
                {
                    "response_id": opaque,
                    "seed": seed,
                    "condition": condition,
                    "family": candidate["family"],
                    "prompt_id": candidate["prompt_id"],
                    "input_prompt_id": (
                        candidate["prompt_id"]
                        if condition == "matched"
                        else episodes[wrong_indices[index]][0]["prompt_id"]
                    ),
                    "output_hex": generated.generated.hex(),
                    "output_text": generated.generated.decode("utf-8", errors="replace"),
                    "generated_bytes": len(generated.generated),
                    "terminated_with_end_marker": generated.generated.endswith(b"<|end|>"),
                }
            )
    return outputs


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError("N1-175 holdout evaluation is write-once")
    if list(REPORTS.glob(".tmp-n175-review-*")):
        raise FileExistsError("an interrupted review bundle exists and requires recovery")
    runs = require_completed_runs()
    migration = json.loads(MIGRATION.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if migration["status"] != "three_seed_parent_migrated_no_training":
        raise ValueError("parent migration receipt differs")
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    episodes, excluded, text_nodes_parsed, selected = load_fixed_holdout()
    if len(episodes) < 2:
        raise ValueError(
            "fewer than two preselected evaluable families remain for fixed prompt mismatch"
        )
    all_outputs = []
    for item in runs:
        all_outputs.extend(generate_for_seed(item["seed"], item["report"], episodes))
    case_rows, key_rows = [], []
    by_family = {candidate["family"]: episode for candidate, episode in episodes}
    for family, episode in by_family.items():
        responses = [row for row in all_outputs if row["family"] == family]
        responses.sort(key=lambda row: hashlib.sha256(row["response_id"].encode()).hexdigest())
        case_id = hashlib.sha256(f"review-case:{family}".encode()).hexdigest()[:16]
        case_rows.append(
            {
                "case_id": case_id,
                "status": "awaiting_adjudication",
                "history": [
                    {"user": user, "assistant": assistant} for user, assistant in episode.history
                ],
                "question": episode.user_input,
                "reference_answer": episode.response,
                "responses": [
                    {"response_id": row["response_id"], "text": row["output_text"]}
                    for row in responses
                ],
                "ratings": {row["response_id"]: None for row in responses},
                "rationale": {row["response_id"]: "" for row in responses},
            }
        )
        for row in responses:
            key_rows.append(
                {
                    "case_id": case_id,
                    "response_id": row["response_id"],
                    "family": family,
                    "prompt_id": row["prompt_id"],
                    "seed": row["seed"],
                    "condition": row["condition"],
                    "input_prompt_id": row["input_prompt_id"],
                    "output_hex": row["output_hex"],
                    "generated_bytes": row["generated_bytes"],
                    "terminated_with_end_marker": row["terminated_with_end_marker"],
                }
            )
    for row in excluded:
        candidate = row["candidate"]
        case_rows.append(
            {
                "case_id": hashlib.sha256(
                    f"review-case:{candidate['family']}".encode()
                ).hexdigest()[:16],
                "status": "not_evaluable_no_replacement",
                "question": "",
                "reference_answer": "",
                "exclusion_reason": row["reason"],
                "responses": [],
                "ratings": {},
                "rationale": {},
            }
        )
    review = {
        "schema": "taiji-n1-plan175-blind-semantic-review-v1",
        "status": "ready_for_blind_human_adjudication",
        "instructions": {
            "rating_scale": {
                "0": "不可读、无关、只套话或核心事实明显错误",
                "1": "部分相关，但未解决核心问题或存在关键错误",
                "2": "直接回答问题且核心内容正确",
            },
            "unverifiable_factual_claim": "在rationale注明，并将rating留空待裁决",
            "scoring": "不要求复现reference的逐字前缀；按问题相关性与正确性评分。响应ID匿名且顺序已打乱。",
        },
        "case_count": len(case_rows),
        "eligible_case_count": len(episodes),
        "excluded_preselected_families_no_replacement": len(excluded),
        "cases": case_rows,
    }
    key = {
        "schema": "taiji-n1-plan175-blind-review-key-v1",
        "responses": key_rows,
    }
    receipt = {
        "schema": "taiji-n1-plan175-v3-holdout-generation-v1",
        "status": "generation_complete_waiting_for_human_review",
        "run_reports": [{"seed": item["seed"], "sha256": item["report_sha256"]} for item in runs],
        "corpus_manifest_sha256": corpus["candidate_manifest_sha256"],
        "selected_holdout_families": len(
            json.loads(PARTITION.read_text(encoding="utf-8"))["holdout_families"]
        ),
        "eligible_review_cases": len(episodes),
        "excluded_no_replacement": len(excluded),
        "expected_outputs": len(episodes) * 2 * len(SEEDS),
        "actual_outputs": len(all_outputs),
        "heldout_text_nodes_parsed": text_nodes_parsed,
        "process_cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.monotonic() - started_wall,
        "can_promote": False,
    }
    staging = Path(tempfile.mkdtemp(prefix=".tmp-n175-review-", dir=REPORTS))
    try:
        for name, payload in (
            ("blind_review.json", review),
            ("blind_review_key.json", key),
            ("generation_receipt.json", receipt),
        ):
            with (staging / name).open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        os.replace(staging, OUTPUT_DIR)
    except BaseException:
        # Preserve the staged bundle for forensic recovery; never reread the sealed face silently.
        raise
    print(
        json.dumps(
            {
                field: receipt[field]
                for field in (
                    "status",
                    "eligible_review_cases",
                    "excluded_no_replacement",
                    "expected_outputs",
                    "actual_outputs",
                    "can_promote",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
