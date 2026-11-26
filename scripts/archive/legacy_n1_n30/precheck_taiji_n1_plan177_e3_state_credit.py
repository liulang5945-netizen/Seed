"""E3 零步验收：状态信用候选的保存/新进程恢复、禁写组、optimizer/RNG/cursor 往返。

零更新训练：只在内存里做 k 步诊断以把 AdamW 状态和 cursor 变得非平凡，然后
①存盘 → 新进程加载 → 同一用例的前向（logits/选择/特征）逐位一致；
②optimizer 状态、torch/python RNG、cursor 一致；
③逐个模块冻结：被冻组参数不变且无梯度，其余可写组仍能移动；
④checkpoint 里没有 autograd 图。

不写 query/holdout/sealed，不训练 producer/V4。上限 8,000 process-CPU 秒 / 6,000 wall 秒。
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

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.internalization import content_digest  # noqa: E402
from taiji.joint_state_consumption_course import E1_ATTRIBUTE, E1_COLORS  # noqa: E402
from taiji.joint_state_credit import (  # noqa: E402
    MODULE_GROUPS,
    CreditCandidateV1,
    CreditCaseV1,
    StateCreditCandidateV1,
    checkpoint_holds_no_graph,
)

CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan177_e3_state_credit_20261029"
CHECKPOINT = CHECKPOINT_ROOT / "credit_step_003.pt"
REPORT = ROOT / "reports/taiji_n1_plan177_e3_state_credit_20261029.json"
DIAGNOSTIC_STEPS = 3
SEED = 0
FEATURE_DIM = 64
CPU_LIMIT = 8000
WALL_LIMIT = 6000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature(*, entity: str, attribute: str, value: str, index: int):
    from taiji.joint_state_consumption import PersistentContentFeatureV1

    return PersistentContentFeatureV1.create(
        entity=entity,
        attribute=attribute,
        value=value,
        source_input_id=f"e3:{index}",
        source_sha256=content_digest({"e3": index, "value": value}),
        turn_index=0,
        revision=1,
        valid_from=0,
        confidence=1.0,
        continuous=[0.0] * FEATURE_DIM,
    )


def case() -> CreditCaseV1:
    facts = (("钥匙", "黄色"), ("杯子", "红色"), ("球", "蓝色"))
    candidates = tuple(
        CreditCandidateV1.from_value(
            feature=_feature(entity=entity, attribute=E1_ATTRIBUTE, value=value, index=index),
            value=value,
        )
        for index, (entity, value) in enumerate(facts)
    )
    return CreditCaseV1(
        case_id="e3:three-entities",
        entity=facts[0][0],
        attribute=E1_ATTRIBUTE,
        candidates=candidates,
        label=E1_COLORS.index(facts[0][1]),
    )


def forward_digest(candidate: StateCreditCandidateV1, item: CreditCaseV1) -> dict[str, Any]:
    out = candidate.forward(item)
    logits = [float(value) for value in out["logits"].detach().tolist()]
    return {
        "logits": logits,
        "logits_digest": content_digest({"logits": logits}),
        "selection_index": int(out["selection_index"]),
        "feature_digest": content_digest(
            {"v": [float(x) for x in out["feature"].detach().tolist()]}
        ),
        "loss": float(out["loss"].detach()),
    }


def restore_state(candidate: StateCreditCandidateV1) -> dict[str, Any]:
    return {
        "optimizer": content_digest(candidate.optimizer.state_dict()),
        "torch_rng": content_digest({"rng": torch.random.get_rng_state().tolist()}),
        "python_rng": content_digest({"state": repr(random.getstate())}),
        "cursor": candidate.cursor,
        "parameters": candidate.parameters_digest(),
    }


def child_restore(path: Path, item: CreditCaseV1) -> dict[str, Any]:
    script = (
        "import json, sys, torch\n"
        "from taiji.joint_state_credit import StateCreditCandidateV1\n"
        "from scripts.training.precheck_taiji_n1_plan177_e3_state_credit import (\n"
        "    case, forward_digest, restore_state,\n"
        ")\n"
        "torch.set_num_threads(1)\n"
        "payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
        "candidate = StateCreditCandidateV1.from_checkpoint(payload)\n"
        "result = {\n"
        "    'format': payload['format'],\n"
        "    'forward': forward_digest(candidate, case()),\n"
        "    'state': restore_state(candidate),\n"
        "    'mode': candidate.mode,\n"
        "}\n"
        "print(json.dumps(result))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1"),
        timeout=1800,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fresh-process restore failed: {completed.stderr[-800:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def forbidden_groups(item: CreditCaseV1) -> list[dict[str, Any]]:
    results = []
    for frozen in MODULE_GROUPS:
        candidate = StateCreditCandidateV1(
            mode="credit_on", seed=SEED, feature_dim=FEATURE_DIM, freeze=(frozen,)
        )
        before = candidate.parameters_digest()
        out = candidate.forward(item)
        gradients = candidate.backward(out["loss"])
        candidate.step()
        after = candidate.parameters_digest()
        moved = {group: before[group] != after[group] for group in MODULE_GROUPS}
        results.append(
            {
                "frozen_group": frozen,
                "frozen_unchanged": moved[frozen] is False,
                "frozen_gradient_zero": gradients[frozen] == 0.0,
                "moved_groups": sorted(group for group, value in moved.items() if value),
                "unmoved_trainable_groups": sorted(
                    group for group in MODULE_GROUPS if group != frozen and not moved[group]
                ),
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177 E3 状态信用零步验收")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.report.exists():
        raise ValueError("E3 state-credit report is write-once")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    item = case()
    candidate = StateCreditCandidateV1(mode="credit_on", seed=SEED, feature_dim=FEATURE_DIM)
    zero_state = restore_state(candidate)
    zero_forward = forward_digest(candidate, item)
    for _ in range(DIAGNOSTIC_STEPS):
        out = candidate.forward(item)
        candidate.backward(out["loss"])
        candidate.step()
    trained_state = restore_state(candidate)
    trained_forward = forward_digest(candidate, item)
    payload = candidate.checkpoint()
    torch.save(payload, args.checkpoint)
    checkpoint_sha256 = _sha256_file(args.checkpoint)

    child = child_restore(args.checkpoint, item)
    parent_after_reload = StateCreditCandidateV1.from_checkpoint(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    )
    second_forward = forward_digest(parent_after_reload, item)

    frozen_results = forbidden_groups(item)
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    verdicts = {
        "checkpoint_written": args.checkpoint.is_file(),
        "checkpoint_sha256_recorded": len(checkpoint_sha256) == 64,
        "checkpoint_holds_no_autograd_graph": checkpoint_holds_no_graph(payload),
        "fresh_process_forward_matches_parent": child["forward"]["logits_digest"]
        == trained_forward["logits_digest"]
        and child["forward"]["selection_index"] == trained_forward["selection_index"]
        and child["forward"]["feature_digest"] == trained_forward["feature_digest"],
        "fresh_process_optimizer_matches": child["state"]["optimizer"]
        == trained_state["optimizer"],
        "fresh_process_rng_matches": child["state"]["torch_rng"] == trained_state["torch_rng"]
        and child["state"]["python_rng"] == trained_state["python_rng"],
        "fresh_process_cursor_matches": child["state"]["cursor"] == trained_state["cursor"],
        "fresh_process_parameters_match": child["state"]["parameters"]
        == trained_state["parameters"],
        "second_forward_matches_first": second_forward == child["forward"],
        "diagnostic_steps_changed_parameters": zero_state["parameters"]
        != trained_state["parameters"],
        "cursor_advanced_by_diagnostics": trained_state["cursor"] == DIAGNOSTIC_STEPS,
        "every_frozen_group_unchanged": all(item["frozen_unchanged"] for item in frozen_results),
        "every_frozen_group_gradient_zero": all(
            item["frozen_gradient_zero"] for item in frozen_results
        ),
        "frozen_encoder_still_lets_readout_move": next(
            item for item in frozen_results if item["frozen_group"] == "encoder"
        )["moved_groups"]
        == ["readout"],
        "frozen_readout_still_lets_encoder_move": next(
            item for item in frozen_results if item["frozen_group"] == "readout"
        )["moved_groups"]
        == ["encoder"],
        "within_cpu_budget": cpu_seconds <= CPU_LIMIT,
        "within_wall_budget": wall_seconds <= WALL_LIMIT,
    }
    report = {
        "format": "taiji-n177-e3-state-credit-v1",
        "date": "2026-10-29",
        "scope": (
            "zero-step engineering acceptance for the E2c bundle: save, fresh-process "
            "restore, forward/optimizer/RNG/cursor equality, per-group freeze; no formal "
            "training, no query/holdout/sealed read"
        ),
        "checkpoint": str(args.checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha256,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "zero_forward": zero_forward,
        "trained_forward": trained_forward,
        "child_forward": child["forward"],
        "trained_state": trained_state,
        "child_state": child["state"],
        "frozen_groups": frozen_results,
        "verdicts": verdicts,
        "all_verdicts_ok": all(verdicts.values()),
        "limits": {"process_cpu_seconds": CPU_LIMIT, "wall_seconds": WALL_LIMIT},
        "cost": {"process_cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds},
    }
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    for key, value in verdicts.items():
        print(f"{key}={value}")
    print(f"ALL_VERDICTS_OK={report['all_verdicts_ok']} out={args.report.name}")
    return 0 if report["all_verdicts_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
