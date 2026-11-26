"""T6：给四份 checkpoint 接上 **CAP D+E 机器计分**（判读线见预注册，先冻结后跑）。

预注册：`plans/reference/M5_R2_T6_CAP_PREREG_20260924.md`。

## 设计

* **M2 计分**：直接调冻结的 `eval_taiji_cap0_baseline.run_baseline(dimensions=("D","E"))`，
  与 M5_R2 主门同口径（默认 `relax_legacy_guard=False` ⇒ 产品链路）。
* **M1 成句率（副读数）**：复用 `eval_taiji_r2_readout_retrain` 的同一套
  `m1_tasks / build_ngram_model / generate / well_formed / assert_criterion_discriminates`，
  但**不进它的 A/B/C 配对门**——`verify_pairing` 校验的是 A/B/C 战役的 trainer/预算/前缀一致，
  本件的 checkpoint 不是那个 trainer 产的；**守卫不删、不放宽，只是不进那个入口**。
* **仪器自检**：`base`（16M）的 D+E 之和必须 = 0（已知 0/36）；否则全部读数作废。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

M2_DIMENSIONS = ("D", "E")

CHECKPOINTS: list[tuple[str, str]] = [
    ("t5", "output/taiji_r2_t5/t5_r0_cue/checkpoint_5000000.pt"),
    ("t1", "output/taiji_r2_t1t2/t1/checkpoint_5000000.pt"),
    ("abl", "output/taiji_r2_t1t2/abl_fabric/checkpoint_5000000.pt"),
    ("base", "checkpoints/seed_beta.pt"),
]


def _cap_d_plus_e(checkpoint: Path) -> dict[str, Any]:
    from eval_taiji_cap0_baseline import run_baseline

    report = run_baseline(checkpoint=checkpoint, dimensions=M2_DIMENSIONS)
    tally = {key: report["dimensions"][key]["tally"] for key in M2_DIMENSIONS}
    return {
        "sum_machine_scored_correct": sum(
            int(tally[key]["machine_scored_correct"] or 0) for key in M2_DIMENSIONS
        ),
        "per_dimension": tally,
        "identity": report.get("identity"),
    }


def _m1_sentence_formation(checkpoint: Path) -> dict[str, Any]:
    """M1 副读数：同一套冻结题面与判据，逐 checkpoint 生成 + 回声控制。"""
    import time

    from diag_taiji_r2_surface_decode import _serialize_prompt
    from eval_taiji_r2_readout_retrain import (
        assert_criterion_discriminates,
        build_ngram_model,
        generate,
        m1_tasks,
        well_formed,
    )

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    taiji = runtime.model.substrate
    tasks, prompt_meta = m1_tasks()
    ngram = build_ngram_model()
    controls = assert_criterion_discriminates(ngram)

    texts: list[str] = []
    started = time.perf_counter()
    for _, prompt in tasks:
        texts.append(
            generate(taiji, _serialize_prompt(runtime, prompt), "greedy").decode(
                "utf-8", errors="replace"
            )
        )
    flags = [1 if well_formed(text, ngram) else 0 for text in texts]
    echoes = [
        _prompt_echo_rate(text, prompt) for text, (_, prompt) in zip(texts, tasks, strict=True)
    ]
    return {
        "n": len(texts),
        "well_formed_rate": round(sum(flags) / len(flags), 4),
        "prompt_echo_8gram_rate": round(sum(echoes) / len(echoes), 4),
        "seconds": round(time.perf_counter() - started, 2),
        "samples": texts[:3],
        "prompt_meta": prompt_meta,
        "controls": controls,
    }


def _prompt_echo_rate(text: str, prompt: str, width: int = 8) -> int:
    from eval_taiji_r2_readout_retrain import _prompt_echo_rate

    return _prompt_echo_rate(text, prompt, width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    scores: dict[str, Any] = {}
    for name, relative in CHECKPOINTS:
        path = Path(relative)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise SystemExit(f"{name} 的 checkpoint 不存在：{path}")
        started = time.perf_counter()
        cap = _cap_d_plus_e(path)
        scores[name] = {
            "checkpoint": str(path),
            "cap_d_plus_e": cap,
            "m1_sentence_formation": _m1_sentence_formation(path),
            "seconds": round(time.perf_counter() - started, 2),
        }
        print(
            json.dumps(
                {
                    "name": name,
                    "cap_d_plus_e": scores[name]["cap_d_plus_e"]["sum_machine_scored_correct"],
                    "m1_well_formed_rate": scores[name]["m1_sentence_formation"][
                        "well_formed_rate"
                    ],
                    "seconds": scores[name]["seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    # §3 仪器自检：base 的 D+E 之和必须 = 0（已知 0/36）
    sanity_ok = scores["base"]["cap_d_plus_e"]["sum_machine_scored_correct"] == 0
    s_t5 = scores["t5"]["cap_d_plus_e"]["sum_machine_scored_correct"]
    s_t1 = scores["t1"]["cap_d_plus_e"]["sum_machine_scored_correct"]
    s_abl = scores["abl"]["cap_d_plus_e"]["sum_machine_scored_correct"]
    s_base = scores["base"]["cap_d_plus_e"]["sum_machine_scored_correct"]

    if not sanity_ok:
        verdict = {
            "verdict": "INSTRUMENT_INVALID",
            "note": "base 的 CAP D+E ≠ 0/36 ⇒ 仪器或口径有问题，全部读数作废",
        }
    elif s_t5 >= 2 and s_t5 > max(s_t1, s_abl, s_base):
        verdict = {"verdict": "SEPARABLE_CUE_BUYS_ANSWERS", "note": "端到端信号成立"}
    elif s_t5 == 0:
        verdict = {
            "verdict": "SEPARABLE_CUE_DOES_NOT_BUY_ANSWERS",
            "note": "下一步才轮到稳定性归因",
        }
    else:
        verdict = {
            "verdict": "SIGNAL_BELOW_MARGIN",
            "note": "有信号但未过 2 题增量门槛；由所有者在复核与稳定性归因之间裁决",
            "scores": {"t5": s_t5, "t1": s_t1, "abl": s_abl, "base": s_base},
        }

    payload = {
        "format": "taiji-r2-cap-checkpoint-scores-v1",
        "prereg": "plans/reference/M5_R2_T6_CAP_PREREG_20260924.md",
        "dimensions": list(M2_DIMENSIONS),
        "sanity_base_d_plus_e_zero": sanity_ok,
        "scores": scores,
        "frozen_summary": {"S_t5": s_t5, "S_t1": s_t1, "S_abl": s_abl, "S_base": s_base},
        "verdict": verdict,
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else (PROJECT_ROOT / "reports" / "taiji_r2_cap_checkpoint_scores_20260924.json")
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(out), "verdict": verdict}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
