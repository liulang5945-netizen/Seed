"""Preserve the actual N175 parent while preparing a restorable V4 zero-update candidate."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from scripts.training.audit_taiji_n1_plan176_learning_paths import (
    CORPUS,
    PARENT,
    PARENT_REPORT,
    credit_and_state_paths,
    digest,
    prompt_state,
)
from scripts.training.build_taiji_n1_plan176_foundation_corpus import ROOT, as_language_episode
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3
from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4
from taiji.persistence import atomic_save

OUTPUT = ROOT / "data/checkpoints/n1_plan176_v4_parent_20261026/tf_seed169/zero.pt"
REPORT = ROOT / "reports/taiji_n1_plan176_v4_parent_migration_20261026.json"


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("V4 migration already exists; preserve its source lineage")
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    receipt = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    parent_sha = digest(PARENT)
    if parent_sha != receipt["final_checkpoint_sha256"]:
        raise ValueError("N175 parent file SHA differs")
    source = torch.load(PARENT, map_location="cpu", weights_only=False)
    parent, _, _ = ConditionalLanguageTrainerV3.from_checkpoint(source)
    trainer, state, inherited_cursor = ConditionalLanguageTrainerV4.from_v3_checkpoint(source)
    old_named, new_named = dict(parent.model.named_parameters()), dict(
        trainer.model.named_parameters()
    )
    for name, parameter in old_named.items():
        if not torch.equal(parameter, new_named[name]):
            raise ValueError(f"shared parameter {name} differs after migration")
        for key, value in parent.optimizer.state[parameter].items():
            if not torch.equal(value, trainer.optimizer.state[new_named[name]][key]):
                raise ValueError(f"shared Adam state {name}/{key} differs")
    cursor = {"stage": "n176-v4-migration-zero", "inherited_n175_cursor": inherited_cursor}
    payload = trainer.checkpoint(state, cursor=cursor)
    atomic_save(payload, OUTPUT)
    code = (
        "import json,sys,time,torch; torch.set_num_threads(1); started=time.process_time(); "
        "from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4; "
        "p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "t,s,c=ConditionalLanguageTrainerV4.from_checkpoint(p); "
        "assert t.checkpoint(s,cursor=c)['checkpoint_digest']==p['checkpoint_digest']; "
        "print(json.dumps({'status':'passed','steps':t.steps,'cpu_seconds':time.process_time()-started}))"
    )
    fresh = subprocess.run(
        [sys.executable, "-c", code, str(OUTPUT)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    restore = json.loads(fresh.stdout)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    train_path = ROOT / corpus["output_directory"] / "train.jsonl"
    if digest(train_path) != corpus["split_sha256"]["train"]:
        raise ValueError("foundation train file differs")
    row = next(
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["metadata"]["query_kind"] == "primary_color"
    )
    episode = as_language_episode(row)
    paths = credit_and_state_paths(trainer.model, episode)
    with torch.no_grad():
        old_logits = parent.model.next_logits(prompt_state(parent.model, episode.prompt_bytes))[0]
        new_logits = trainer.model.next_logits(prompt_state(trainer.model, episode.prompt_bytes))[0]
        delta = float((old_logits - new_logits).abs().max())
    if (
        paths["task_loss_gradient_norms"]["state.workspace"] is None
        or paths["task_loss_gradient_norms"]["state.workspace"] <= 0
        or paths["state_perturbation_max_logit_delta"]["workspace"] <= 0
        or trainer.checkpoint(state, cursor=cursor)["checkpoint_digest"]
        != payload["checkpoint_digest"]
        or digest(PARENT) != parent_sha
    ):
        raise ValueError("V4 workspace path or checkpoint immutability check failed")
    report = {
        "schema": "taiji-n176-v3-to-v4-parent-migration-v1",
        "status": "actual_parent_migrated_restored_workspace_consumed_no_updates",
        "parent_checkpoint": str(PARENT.relative_to(ROOT)),
        "parent_checkpoint_sha256": parent_sha,
        "checkpoint": str(OUTPUT.relative_to(ROOT)),
        "checkpoint_sha256": digest(OUTPUT),
        "checkpoint_digest": payload["checkpoint_digest"],
        "candidate_source_sha256": digest(ROOT / "taiji/conditional_language_v4.py"),
        "migration_source_sha256": digest(Path(__file__)),
        "shared_parameters_exact": len(old_named),
        "shared_adam_states_exact": len(parent.optimizer.state),
        "parent_parameter_count": sum(p.numel() for p in old_named.values()),
        "candidate_parameter_count": sum(p.numel() for p in new_named.values()),
        "inherited_steps": trainer.steps,
        "inherited_target_byte_exposures": trainer.target_byte_exposures,
        "fresh_process_restore": restore,
        "train_probe_episode": row["episode_id"],
        "train_probe_paths": paths,
        "initial_parent_max_logit_delta_on_one_train_prompt": delta,
        "parent_logits_identical": False,
        "training_updates": 0,
        "new_sealed_text_read": 0,
        "n175_key_opened": False,
        "process_cpu_seconds": time.process_time() - started_cpu,
        "child_process_cpu_seconds": restore["cpu_seconds"],
        "wall_seconds": time.monotonic() - started_wall,
        "can_promote": False,
        "limitations": "Engineering reachability only; no foundation learning or semantic gain measured. The unchanged V3 state schema stores neural evidence chunks, not proven object-level world knowledge.",
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
                    "candidate_parameter_count",
                    "inherited_steps",
                    "process_cpu_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
