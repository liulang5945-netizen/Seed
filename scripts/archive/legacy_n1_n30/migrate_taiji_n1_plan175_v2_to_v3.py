"""Create durable N1-175 V3 continuations from the completed N1-174 TF seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest
from taiji.conditional_language_v3 import (
    ConditionalLanguageTrainerV3,
    TaskCoreConfigV3,
)

SEEDS = (169, 170, 171)
PARENT_REPORTS = ROOT / "reports"
PARENT_CHECKPOINTS = ROOT / "data/checkpoints/n1_plan174c_scale_complete_20261025"
DESTINATION = ROOT / "data/checkpoints/n1_plan175_v3_parent_20261026"
REPORT = ROOT / "reports/taiji_n1_plan175_v3_parent_migration_20261026.json"


def sha256(path: Path) -> str:
    return digest(path)


def verify_fresh_process(path: Path, expected_digest: str, expected_parent_sha: str) -> float:
    code = (
        "import json,torch,sys,time; started=time.process_time(); "
        "from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3; "
        "p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "t,state,c=ConditionalLanguageTrainerV3.from_checkpoint(p); "
        "assert p['checkpoint_digest']==sys.argv[2]; "
        "assert t.lineage['parent_checkpoint_sha256']==sys.argv[3]; "
        "assert t.steps==5000 and c['epoch']==0 and c['batch']==0; "
        "assert state.to_payload()['format']=='taiji-task-state-v3'; "
        "print(json.dumps({'digest':p['checkpoint_digest'],'cpu_seconds':time.process_time()-started}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), expected_digest, expected_parent_sha],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        check=True,
        text=True,
    )
    restored = json.loads(result.stdout)
    if restored["digest"] != expected_digest:
        raise ValueError("fresh-process V3 checkpoint digest differs")
    return float(restored["cpu_seconds"])


def migrate() -> dict:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    outputs = [DESTINATION / f"tf_seed{seed}" / "zero.pt" for seed in SEEDS]
    started = time.process_time()
    wall_started = time.monotonic()
    rows = []
    child_cpu = 0.0
    for seed, destination in zip(SEEDS, outputs, strict=True):
        source_report_path = (
            PARENT_REPORTS / f"taiji_n1_plan174c_scale_complete_20261025_tf_seed{seed}.json"
        )
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        if source_report.get("status") != "complete" or source_report.get("seed") != seed:
            raise ValueError(f"N1-174C seed {seed} is not an authoritative complete parent")
        final = source_report["restores"]["final"]
        source_path = Path(final["path"])
        source_sha = sha256(source_path)
        if source_sha != final["sha256"]:
            raise ValueError(f"N1-174C seed {seed} parent SHA differs from its report")
        parent = torch.load(source_path, map_location="cpu", weights_only=False)
        if parent.get("checkpoint_digest") != final["checkpoint_digest"]:
            raise ValueError(f"N1-174C seed {seed} checkpoint digest differs from its report")
        config = TaskCoreConfigV3(
            width=parent["config"]["width"],
            embedding_width=parent["config"]["embedding_width"],
            key_width=parent["config"]["key_width"],
            slots=parent["config"]["slots"],
            chunk_bytes=parent["config"]["chunk_bytes"],
            max_observation_bytes=4096,
            max_input_evidence_bytes=4096,
            max_output_position=2048,
            max_generation_bytes=2048,
            seed=parent["config"]["seed"],
            position_extension_seed=seed + 1,
        )
        if destination.exists():
            payload = torch.load(destination, map_location="cpu", weights_only=False)
            trainer, _, _ = ConditionalLanguageTrainerV3.from_checkpoint(payload)
            if (
                trainer.lineage.get("parent_checkpoint_sha256") != source_sha
                or trainer.model.config.position_extension_seed != seed + 1
            ):
                raise ValueError(f"existing seed {seed} V3 checkpoint has unrelated lineage/config")
        else:
            trainer = ConditionalLanguageTrainerV3.from_v2_checkpoint(
                parent,
                config=config,
                parent_checkpoint_sha256=source_sha,
            )
            state = trainer.model.new_state(
                scope_id="n1-175", session_id=f"seed-{seed}", native_state_ref=f"n1-175:seed-{seed}"
            )
            cursor = {"epoch": 0, "batch": 0, "parent_cursor": final["cursor"]}
            trainer.save(destination, state, cursor=cursor)
            payload = torch.load(destination, map_location="cpu", weights_only=False)
        target_sha = sha256(destination)
        child_cpu += verify_fresh_process(destination, payload["checkpoint_digest"], source_sha)
        rows.append(
            {
                "seed": seed,
                "parent_checkpoint": str(source_path.relative_to(ROOT)),
                "parent_checkpoint_sha256": source_sha,
                "parent_checkpoint_digest": final["checkpoint_digest"],
                "parent_steps": parent["steps"],
                "parent_cursor": final["cursor"],
                "v3_checkpoint": str(destination.relative_to(ROOT)),
                "v3_checkpoint_sha256": target_sha,
                "v3_checkpoint_digest": payload["checkpoint_digest"],
                "v3_parameter_count": sum(
                    value.numel() for value in payload["parameters"].values()
                ),
                "optimizer_state_parameter_count": len(payload["optimizer"]["state"]),
                "position_rows_copied": trainer.lineage["position_rows_copied"],
                "fresh_process_restore": "passed",
            }
        )
    return {
        "schema": "taiji-n1-plan175-v2-to-v3-parent-migration-v1",
        "status": "three_seed_parent_migrated_no_training",
        "migration_script_sha256": sha256(Path(__file__)),
        "v3_candidate_sha256": sha256(ROOT / "taiji/conditional_language_v3.py"),
        "parents": rows,
        "training_updates": 0,
        "new_holdout_text_read": 0,
        "cpu_seconds": time.process_time() - started + child_cpu,
        "wall_seconds": time.monotonic() - wall_started,
        "can_promote": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-checkpoint", type=Path)
    args = parser.parse_args()
    if args.verify_checkpoint is not None:
        payload = torch.load(args.verify_checkpoint, map_location="cpu", weights_only=False)
        trainer, state, cursor = ConditionalLanguageTrainerV3.from_checkpoint(payload)
        print(
            json.dumps(
                {
                    "checkpoint_digest": payload["checkpoint_digest"],
                    "steps": trainer.steps,
                    "cursor": cursor,
                    "state_format": state.to_payload()["format"],
                }
            )
        )
        return
    payload = migrate()
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cpu_seconds": payload["cpu_seconds"],
                "parents": len(payload["parents"]),
            }
        )
    )


if __name__ == "__main__":
    main()
