"""门槛① 在**装机形态**的真实 `chat()` 链上到底拦下多少答复（产品入口，不重抄生成链）。

为什么要这一档：`api/seed_runtime.py:500-512` 的回写门槛只在 `gate_model is not None` 时生效
（＝回路在场且门槛 `armed`），而 §2v 把"回写通道"算成回路的第 3 项代价。
可是**没人量过它在装机面上实际拦不拦**——不拦的话"只回写过门答复"这句就是纸面承诺。

链路与其它表层件**完全同一条**：`SeedRuntime.chat(..., learn=True)`，题面取自
`score_taiji_r2_copy_surface_extension.load_items`（同一份 manifest，**不另造题面、不改生成链**）。
判定直接读产品自己的披露字段 `runtime.last_write_back_gate`，不在这里重算放行逻辑。

判读线（**先于数冻结**，四支互斥）：
* 门槛没武装／回路不在场 ⇒ `not_measurable`（不许把"门没跑"读成"门全放行"以外的任何结论）；
* `allowed_rate == 1.0` ⇒ `gate_never_blocks_on_this_face`；
* `allowed_rate <= 0.5` ⇒ `gate_blocks_majority`；
* 其余 ⇒ `gate_blocks_some`。

写盘前后各取一次检查点 sha：本件只读，**落盘守卫必须为 true**（`chat(learn=True)` 只动内存态）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from api.seed_runtime import SeedRuntime  # noqa: E402
from score_taiji_r2_copy_surface_extension import load_items  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "checkpoints/seed_beta_with_circuit.pt"
ROUNDS_PER_ITEM = 3
ITEM_LIMIT = 24
OUT = PROJECT_ROOT / "reports/taiji_a30_writeback_gate_on_shipping_face_20260930.json"


def decide(gate_armed: bool, circuit_present: bool, calls: int, allowed: int) -> str:
    """四支互斥；**缺 armed 就响亮不判**（零拦截与"没门"是两件事）。"""

    if calls == 0:
        return "not_measured（一次调用都没发生，任何率都不许报）"
    if not (gate_armed and circuit_present):
        return "not_measurable（门槛未武装或回路不在场 ⇒ 产品本就不跑这条门）"
    rate = allowed / calls
    if rate == 1.0:
        return "gate_never_blocks_on_this_face"
    if rate <= 0.5:
        return "gate_blocks_majority"
    return "gate_blocks_some"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # GBK 控制台会在打印中文样例时崩
        sys.stdout.reconfigure(encoding="utf-8")

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=ITEM_LIMIT, help="取题集前 N 条（冒烟用小值）")
    parser.add_argument("--rounds", type=int, default=ROUNDS_PER_ITEM, help="每条题面走几轮")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()
    item_limit, rounds = args.items, args.rounds
    out = Path(args.out_report) if args.out_report else OUT

    sha_before = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    runtime = SeedRuntime.load(CHECKPOINT)
    items = load_items()[:item_limit]

    reasons: dict[str, int] = {}
    calls = 0
    allowed = 0
    samples: list[str] = []
    for item in items:
        history: list[tuple[str, str]] = []
        for turn in [str(t) for t in item["turns"][:rounds]]:
            answer = runtime.chat(turn, history=history, learn=True)
            state = runtime.last_write_back_gate
            calls += 1
            if state is None:
                reasons["no_gate_state"] = reasons.get("no_gate_state", 0) + 1
            else:
                ok, reason = state
                allowed += int(bool(ok))
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
                if not ok and len(samples) < 3:
                    samples.append(str(turn)[:24])
            # 与产品用法一致：下一轮的 prompt 带上当前的问／答（历史由调用方给）。
            history.append((turn, str(answer)))

    sha_after = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    circuit_present = getattr(runtime.model.substrate, "copy_circuit", None) is not None
    gate_armed = runtime.surface_gate_state == "armed"

    report = {
        "format": 1,
        "question": "装机形态（默认件自动挂回路）下，门槛① 在真实 chat 链上拦下多少答复",
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "manifest": "plans/manifests/r2_copy_surface_extension_v1.json（同表层件那份）",
        "chain": "SeedRuntime.chat(learn=True)，判定读产品披露字段 last_write_back_gate",
        "items": len(items),
        "calls": calls,
        "allowed": allowed,
        "blocked": calls - allowed,
        "allowed_rate": round(allowed / calls, 4) if calls else None,
        "reason_histogram": dict(sorted(reasons.items(), key=lambda pair: -pair[1])),
        "blocked_question_samples": samples,
        "assembly": {
            "surface_gate_state": runtime.surface_gate_state,
            "circuit_present": circuit_present,
            "write_back_gate_last": (
                [bool(runtime.last_write_back_gate[0]), str(runtime.last_write_back_gate[1])]
                if runtime.last_write_back_gate
                else None
            ),
        },
        "instrument_guard": {
            "checkpoint_untouched": sha_before == sha_after,
            "expected_calls": len(items) * rounds,
            "calls_match_expected": calls == len(items) * rounds,
            "no_gate_state_rows": reasons.get("no_gate_state", 0) == 0,
        },
        "sha256_before": sha_before[:16],
        "verdict": decide(gate_armed, circuit_present, calls, allowed),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True)[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
