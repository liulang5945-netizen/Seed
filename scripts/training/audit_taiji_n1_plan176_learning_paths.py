"""Audit existing V3 information and credit paths on train-only foundation tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.training.build_taiji_n1_plan176_foundation_corpus import (
    COLORS,
    OBJECTS,
    OWNERS,
    PLACES,
    ROOT,
    as_language_episode,
    pair_split,
)
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3

REPORT = ROOT / "reports/taiji_n1_plan176_learning_paths_20261026.json"
CORPUS = ROOT / "reports/taiji_n1_plan176_foundation_corpus_20261026.json"
PARENT = ROOT / "data/checkpoints/n1_plan175_v3_scale_20261026/tf_seed169/epoch_40.pt"
PARENT_REPORT = ROOT / "reports/taiji_n1_plan175_v3_scale_20261026_tf_seed169.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alternative_color(pair: list[str]) -> str:
    oi = OBJECTS.index(pair[0])
    return next(c for i, c in enumerate(COLORS) if c != pair[1] and pair_split(oi, i) == "train")


def interventions(row: dict) -> tuple[dict, dict]:
    """Change only the queried fact, or only a nonqueried color fact."""
    related, irrelevant = copy.deepcopy(row), copy.deepcopy(row)
    kind = row["metadata"]["query_kind"]
    if kind in {"primary_color", "distractor_color"}:
        field = "primary_pair" if kind == "primary_color" else "distractor_pair"
        obj, color = row["metadata"][field]
        new = alternative_color([obj, color])
        old_clause, new_clause = f"{obj}是{color}的。", f"{obj}是{new}的。"
        related["metadata"][field][1] = new
    elif kind == "location_update":
        old = row["metadata"]["final_location"]
        new = next(p for p in PLACES if p not in {old, row["metadata"]["initial_location"]})
        old_clause, new_clause = f"移到了{old}。", f"移到了{new}。"
        related["metadata"]["final_location"] = new
    elif kind == "owner_correction":
        metadata = row["metadata"]
        old, obj = metadata["final_owner"], metadata["primary_pair"][0]
        owner = next(p for p in OWNERS if p not in {old, metadata["initial_owner"]})
        new = owner + "的"
        old_clause, new_clause = f"更正一下，{obj}是{old}的", f"更正一下，{obj}是{owner}的"
        related["metadata"]["final_owner"] = owner
    else:
        raise ValueError("only F1/F2 semantic tasks have these interventions")
    if related["user_input"].count(old_clause) != 1:
        raise ValueError("related fact is not uniquely identified")
    related["user_input"] = related["user_input"].replace(old_clause, new_clause, 1)
    related["metadata"]["answer_from_world"] = new
    related["response"] = new + "。"
    other = "primary_pair" if kind == "distractor_color" else "distractor_pair"
    obj, color = row["metadata"][other]
    alternative = alternative_color([obj, color])
    old_clause, new_clause = f"{obj}是{color}的。", f"{obj}是{alternative}的。"
    if irrelevant["user_input"].count(old_clause) != 1:
        raise ValueError("irrelevant fact is not uniquely identified")
    irrelevant["user_input"] = irrelevant["user_input"].replace(old_clause, new_clause, 1)
    irrelevant["metadata"][other][1] = alternative
    return related, irrelevant


def prompt_state(model, prompt: bytes):
    state = model.new_state(
        scope_id="n176-audit", session_id="train", native_state_ref="n176:audit"
    )
    return model.step_task(
        prompt,
        state,
        scope_id=state.scope_id,
        session_id=state.session_id,
        source_id="prompt",
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 4096},
    )


def trace(model, state, target: bytes):
    cache = model.evidence_cache(state)
    logits = []
    for symbol in target:
        current, hidden = model.next_logits(state, cache=cache)
        logits.append(current)
        state = model.observe_response(state, symbol, hidden)
    return torch.stack(logits)


def normalized_answer(raw: bytes) -> str | None:
    if raw.endswith(b"<|end|>"):
        raw = raw[: -len(b"<|end|>")]
    try:
        return raw.decode("utf-8", errors="strict").strip().rstrip("。.!！")
    except UnicodeDecodeError:
        return None


def measure(model, prompt: bytes, fixed_target: bytes, actual_target: bytes) -> dict:
    with torch.no_grad():
        state = prompt_state(model, prompt)
        logits = trace(model, state, fixed_target)
        target = torch.tensor(list(fixed_target), dtype=torch.long)
        generated = model.generate(prompt_state(model, prompt), max_new_bytes=96).generated
    return {
        "fixed_original_target_nll": float(F.cross_entropy(logits, target)),
        "fixed_original_target_tf_top1": float((logits.argmax(-1) == target).float().mean()),
        "free_correct_for_actual_fact": normalized_answer(generated)
        == normalized_answer(actual_target),
        "generated_text": generated.decode("utf-8", errors="replace"),
        "generated_bytes": len(generated),
        "terminated": generated.endswith(b"<|end|>"),
    }


def credit_and_state_paths(model, episode) -> dict:
    state = prompt_state(model, episode.prompt_bytes)
    logits = trace(model, state, episode.target_bytes)
    loss = F.cross_entropy(logits, torch.tensor(list(episode.target_bytes), dtype=torch.long))
    parameters = list(model.named_parameters())
    nodes = [state.workspace, state.byte_memory, state.hidden]
    gradients = torch.autograd.grad(loss, nodes + [p for _, p in parameters], allow_unused=True)
    names = ["state.workspace", "state.byte_memory", "state.hidden"] + [n for n, _ in parameters]
    norms = {
        n: None if g is None else float(g.norm()) for n, g in zip(names, gradients, strict=True)
    }
    with torch.no_grad():
        state = prompt_state(model, episode.prompt_bytes)
        base = model.next_logits(state)[0]
        perturbations = {}
        for name, amount in (("workspace", 7.0), ("byte_memory", 0.125), ("hidden", 0.125)):
            changed = replace(state, **{name: getattr(state, name) + amount})
            perturbations[name] = float((model.next_logits(changed)[0] - base).abs().max())
    return {"task_loss_gradient_norms": norms, "state_perturbation_max_logit_delta": perturbations}


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("preserve the existing train-only path audit")
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    train = ROOT / corpus["output_directory"] / "train.jsonl"
    parent_report = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    if (
        digest(train) != corpus["split_sha256"]["train"]
        or digest(PARENT) != parent_report["final_checkpoint_sha256"]
    ):
        raise ValueError("frozen train corpus or parent checkpoint differs")
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    kinds = ("primary_color", "distractor_color", "location_update", "owner_correction")
    selected = []
    for kind in kinds:
        pool = sorted(
            (r for r in rows if r["metadata"]["query_kind"] == kind),
            key=lambda r: hashlib.sha256(("n176-path-v1:" + r["episode_id"]).encode()).hexdigest(),
        )
        seen = set()
        for row in pool:
            pair = tuple(row["metadata"]["primary_pair"])
            if pair not in seen:
                seen.add(pair)
                selected.append(row)
                if len(seen) == 2:
                    break
    trainer, original_state, cursor = ConditionalLanguageTrainerV3.from_checkpoint(
        torch.load(PARENT, map_location="cpu", weights_only=False)
    )
    before = trainer.checkpoint(original_state, cursor=cursor)["checkpoint_digest"]
    results = []
    for row in selected:
        if time.process_time() - started_cpu > 120:
            raise TimeoutError("train-only path audit CPU guard reached")
        episode = as_language_episode(row)
        related, irrelevant = (as_language_episode(x) for x in interventions(row))
        peers = [
            r for r in selected if r["metadata"]["query_kind"] == row["metadata"]["query_kind"]
        ]
        peer = next(r for r in peers if r["family_id"] != row["family_id"])
        conditions = {}
        for name, prompt, target in (
            ("matched", episode.prompt_bytes, episode.target_bytes),
            ("related_fact_flip", related.prompt_bytes, related.target_bytes),
            ("irrelevant_fact_flip", irrelevant.prompt_bytes, episode.target_bytes),
            ("swapped_prompt", as_language_episode(peer).prompt_bytes, episode.target_bytes),
        ):
            conditions[name] = measure(trainer.model, prompt, episode.target_bytes, target)
        results.append(
            {
                "episode_id": row["episode_id"],
                "query_kind": row["metadata"]["query_kind"],
                "conditions": conditions,
                "paths": credit_and_state_paths(trainer.model, episode),
            }
        )
    after = trainer.checkpoint(original_state, cursor=cursor)["checkpoint_digest"]
    if before != after:
        raise ValueError("audit changed the parent checkpoint")
    report = {
        "schema": "taiji-n176-information-credit-path-audit-v1",
        "status": "complete_train_only_no_updates",
        "parent_seed": 169,
        "parent_checkpoint_sha256": digest(PARENT),
        "train_corpus_sha256": digest(train),
        "audit_source_sha256": digest(Path(__file__)),
        "selected_episodes": [r["episode_id"] for r in selected],
        "results": results,
        "base_free_correct": sum(
            r["conditions"]["matched"]["free_correct_for_actual_fact"] for r in results
        ),
        "workspace_independent_path_absent": all(
            r["paths"]["task_loss_gradient_norms"]["state.workspace"] is None
            and r["paths"]["state_perturbation_max_logit_delta"]["workspace"] == 0.0
            for r in results
        ),
        "checkpoint_unchanged": before == after,
        "training_updates": 0,
        "new_sealed_text_read": 0,
        "n175_blind_key_opened": False,
        "process_cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.monotonic() - started_wall,
        "limitations": "One N175 parent before foundation training; gradients/logit deltas do not prove semantic use or learning benefit. Swapped-prompt correctness is not a semantic gate; its fixed-target NLL controls target difficulty only.",
    }
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "base_free_correct",
                    "workspace_independent_path_absent",
                    "process_cpu_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
