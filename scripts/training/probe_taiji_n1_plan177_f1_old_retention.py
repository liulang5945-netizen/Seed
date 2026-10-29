"""N177 F1 旧保持探针：N176 F0 的 16 题，在"父 V4"与"F1 候选（带零特征偏置）"上各生成一次。

这是 F 合同把"旧保持下降 ≤3pp"写进 PASS 合取项所需要的执行面（DEBT-G185）。
两个对照都要报：

* `no_bias`：同一 V4、不加偏置 ⇒ 必须与父端**逐字节相同**（证明 V4 没被动过）；
* `with_zero_bias`：候选在无状态输入上的真实行为（偏置 = bias(0)）⇒ 这一支才可能掉分。

不读 query/holdout；只物化 N176 的 16 条 F0 retention episode（train 池内，原登记在
`reports/taiji_n1_plan176_pilot_selection_20261026.json`）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.audit_taiji_n1_plan176_learning_paths import (  # noqa: E402
    normalized_answer,
    prompt_state,
)
from scripts.training.build_taiji_n1_plan176_foundation_corpus import (  # noqa: E402
    as_language_episode,
    build_rows,
)
from scripts.training.run_taiji_n1_plan177_f_f0b import _Arm, _rel  # noqa: E402
from taiji.joint_t1_trace_cache import JointT1TraceCacheV1  # noqa: E402
from taiji.joint_t1_trainer import JointT1TrainerV1  # noqa: E402

PARENT = ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
SELECTION = ROOT / "reports/taiji_n1_plan176_pilot_selection_20261026.json"
F1_ROOT = ROOT / "data/checkpoints/n1_plan177_f_f1_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_f1_old_retention_20261029.json"
MAX_NEW_BYTES = 96
MAX_DROP_PP = 3.0


def _rows() -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in build_rows()}


def _parent_model() -> Any:
    cache = JointT1TraceCacheV1.from_payload(
        torch.load(TRACE_CACHE, map_location="cpu", weights_only=False)
    )
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=cache)
    trainer.input_trace_cache = None
    return trainer.model


def _endpoint(arm: str, seed: int, root: Path = F1_ROOT) -> Path:
    run_root = root / f"{arm}__seed_{seed}"
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    return ROOT / state["endpoint"]


def measure(
    arm_name: str,
    seed: int,
    rows: dict[str, dict[str, Any]],
    ids: list[str],
    model: Any,
    run_root: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint or _endpoint(arm_name, seed, run_root)
    arm = _Arm.from_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False))
    zero_feature = torch.zeros(arm.feature_dim)
    with torch.no_grad():
        bias = arm.bias(zero_feature)
    per_row = []
    base_hits = cand_hits = no_bias_identical = 0
    for identity in ids:
        row = rows[identity]
        episode = as_language_episode(row)
        target = normalized_answer(episode.target_bytes)
        with torch.no_grad():
            base = model.generate(
                prompt_state(model, episode.prompt_bytes), max_new_bytes=MAX_NEW_BYTES
            ).generated
            plain = model.generate(
                prompt_state(model, episode.prompt_bytes),
                max_new_bytes=MAX_NEW_BYTES,
                logit_bias=bias,
            ).generated
        base_correct = normalized_answer(base) == target
        cand_correct = normalized_answer(plain) == target
        base_hits += base_correct
        cand_hits += cand_correct
        no_bias_identical += plain == base
        per_row.append(
            {
                "episode_id": identity,
                "parent_correct": base_correct,
                "candidate_correct": cand_correct,
                "unchanged_bytes": plain == base,
            }
        )
    drop_pp = (base_hits - cand_hits) / len(ids) * 100.0
    return {
        "arm": arm_name,
        "seed": seed,
        "endpoint": _rel(checkpoint or _endpoint(arm_name, seed, run_root)),
        "items": len(ids),
        "parent_hits": base_hits,
        "candidate_hits": cand_hits,
        "drop_pp": drop_pp,
        "within_3pp": drop_pp <= MAX_DROP_PP,
        "rows_where_bias_changed_nothing": no_bias_identical,
        "per_row": per_row,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177 F1 旧保持探针")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--arm", default=None, help="只测一臂（冒烟用）；默认全部 12 个 run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-root", type=Path, default=F1_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重算同一探针件（只允许在本轮内重跑；跨轮仍应写一次）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="冒烟用：直接指定一个 F1 臂 checkpoint（不查 run_state）",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.report.exists() and not args.overwrite:
        raise ValueError("old-retention report is write-once")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    ids = list(selection["F0_retention_episode_ids"])
    rows = _rows()
    missing = [identity for identity in ids if identity not in rows]
    if missing:
        raise RuntimeError(f"F0 retention ids missing from the corpus: {missing}")
    runs = (
        [(args.arm, args.seed)]
        if args.arm is not None and args.seed is not None
        else [
            (arm, seed)
            for seed in (0, 1, 2)
            for arm in ("main_credit_on", "main_credit_stop", "copy_only", "no_state")
        ]
    )
    cpu_start = time.process_time()
    model = _parent_model()
    results = [
        measure(arm, seed, rows, ids, model, args.run_root, args.checkpoint) for arm, seed in runs
    ]
    main_results = [item for item in results if item["arm"] == "main_credit_on"]
    main_worst = max((item["drop_pp"] for item in main_results), default=None)
    report = {
        "format": "taiji-n177-f1-old-retention-v1",
        "date": "2026-10-29",
        "scope": "N176 F0 16 retention episodes; parent V4 vs F1 candidate with bias(0)",
        "max_drop_pp": MAX_DROP_PP,
        "runs": results,
        "worst_drop_pp": max(item["drop_pp"] for item in results),
        "main_worst_drop_pp": main_worst,
        "main_within_3pp": main_worst is not None and main_worst <= MAX_DROP_PP,
        "all_within_3pp": all(item["within_3pp"] for item in results),
        "ruler_resolution_pp": 100.0 / len(ids),
        "cost": {"process_cpu_seconds": time.process_time() - cpu_start},
    }
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    for item in results:
        print(
            f"{item['arm']}__seed_{item['seed']}: parent={item['parent_hits']}/{item['items']} "
            f"candidate={item['candidate_hits']}/{item['items']} drop={item['drop_pp']:.2f}pp "
            f"bias_changed_nothing={item['rows_where_bias_changed_nothing']}"
        )
    print(f"ALL_WITHIN_3PP={report['all_within_3pp']}")
    return 0 if report["all_within_3pp"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
