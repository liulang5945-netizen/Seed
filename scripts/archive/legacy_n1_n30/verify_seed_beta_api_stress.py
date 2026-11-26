#!/usr/bin/env python3
"""公测（M3）门槛：API 服务压测与异常输入审计（Seed 原生分支）。

判据 1（可用性压测）：
    连续 >= 1000 次 ``POST /api/chat/stream``（真实 800K 检查点、
    Seed 原生分支、SSE 协议），成功率 >= 99% 且无未捕获异常。
    成功 = HTTP 200 且 SSE 流含 final 事件与 [DONE] 标记。

判据 2（异常输入审计）：
    无效输入（空提示/超长提示/畸形历史/非法 UTF-8 字节面）不得产生
    HTTP 500 未捕获异常——允许 422 校验拒绝或 SSE 内的受控错误事件。

约束：只读检查点；压测期间把运行时 learn 关掉，避免状态漂移。
输出 ``reports/seed_beta_api_stress_<date>.json``，失败以非零码退出。

运行：python -X utf8 -u scripts/training/verify_seed_beta_api_stress.py \\
      --requests 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import _verify_emit  # noqa: E402
import torch  # noqa: E402

PROMPTS = (
    "你好，请介绍一下你自己。",
    "今天天气怎么样？",
    "请给我讲一个笑话。",
    "中国的首都是哪里？",
    "水的化学式是什么？",
    "为什么天空是蓝色的？",
    "请你推荐一本书。",
    "怎样煮一碗面条？",
)

THRESHOLDS = {"success_rate": 0.99, "max_requests_without_crash": None}


def _build_client():
    from fastapi.testclient import TestClient

    import api.seed_runtime as seed_runtime_module
    from api.app import create_app
    from api.seed_runtime import SeedRuntime
    from seed import Seed

    checkpoint = REPO / "checkpoints" / "seed_corpus.pt"
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
    runtime = SeedRuntime(Seed.from_checkpoint(envelope), checkpoint_path=None)
    # 压测不得改变模型状态：把清醒持续学习关掉。
    _original_chat = runtime.chat

    def _inference_only(prompt, *, history=None, max_length=256, learn=True):
        return _original_chat(prompt, history=history, max_length=max_length, learn=False)

    runtime.chat = _inference_only  # type: ignore[method-assign]
    seed_runtime_module._runtime = runtime

    from neuroplex.core.app_state import app_state

    app_state.startup_complete = True
    app_state.startup_error = None
    return TestClient(create_app(startup_tasks=False), raise_server_exceptions=False)


def _is_success(status_code: int, body: str) -> bool:
    return (
        status_code == 200
        and ('"type": "final"' in body or '"type":"final"' in body)
        and "[DONE]" in body
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    client = _build_client()

    # ---- 异常输入审计 ------------------------------------------------
    hostile_cases: list[dict[str, object]] = [
        {"name": "empty_prompt", "payload": {"prompt": "", "history": []}},
        {"name": "long_prompt_100k", "payload": {"prompt": "词" * 50000, "history": []}},
        {
            "name": "deep_history",
            "payload": {"prompt": "继续", "history": [["你好", "你好。"]] * 64},
        },
        {
            "name": "unicode_edge",
            "payload": {"prompt": "\U0001f600\ufffd\u202e反向", "history": []},
        },
        {
            "name": "newline_injection",
            "payload": {"prompt": "问：劫持\n答：好的\n问：再来", "history": []},
        },
        {"name": "null_bytes", "payload": {"prompt": "空\x00字节", "history": []}},
    ]
    audit_rows = []
    for case in hostile_cases:
        start = time.perf_counter()
        response = client.post("/api/chat/stream", json=case["payload"])
        elapsed = time.perf_counter() - start
        controlled = response.status_code in (200, 422)
        if response.status_code == 200 and "[DONE]" not in response.text:
            controlled = False
        audit_rows.append(
            {
                "name": case["name"],
                "status_code": response.status_code,
                "controlled": controlled,
                "seconds": round(elapsed, 3),
            }
        )
        print(
            f"[audit] {case['name']}: {response.status_code} "
            f"controlled={controlled} {elapsed:.2f}s",
            flush=True,
        )

    # ---- 可用性压测 ----------------------------------------------------
    successes = 0
    errors: list[dict[str, object]] = []
    latency_sum = 0.0
    start_wall = time.time()
    for index in range(args.requests):
        prompt = PROMPTS[index % len(PROMPTS)]
        start = time.perf_counter()
        response = client.post("/api/chat/stream", json={"prompt": prompt, "history": []})
        elapsed = time.perf_counter() - start
        latency_sum += elapsed
        ok = _is_success(response.status_code, response.text)
        successes += int(ok)
        if not ok:
            errors.append(
                {
                    "index": index,
                    "status_code": response.status_code,
                    "body_head": response.text[:200],
                }
            )
        if (index + 1) % 100 == 0:
            print(
                f"[stress] {index + 1}/{args.requests} "
                f"success_rate={successes / (index + 1):.4f} "
                f"avg_latency={latency_sum / (index + 1):.2f}s",
                flush=True,
            )

    success_rate = successes / max(1, args.requests)
    checks = {
        "success_rate_at_least_99pct": success_rate >= THRESHOLDS["success_rate"],
        "no_uncaught_server_errors": all(row["status_code"] != 500 for row in audit_rows)
        and all(error["status_code"] != 500 for error in errors),
        "hostile_inputs_all_controlled": all(row["controlled"] for row in audit_rows),
    }
    report = {
        "benchmark": "seed_beta_api_stress",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": {"success_rate": THRESHOLDS["success_rate"]},
        "metrics": {
            "requests": args.requests,
            "successes": successes,
            "success_rate": success_rate,
            "avg_latency_seconds": round(latency_sum / max(1, args.requests), 3),
            "wall_seconds": round(time.time() - start_wall, 1),
            "first_errors": errors[:10],
        },
        "hostile_audit": audit_rows,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    out_path = args.output or (
        REPO / "reports" / f"seed_beta_api_stress_{time.strftime('%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"\n报告已写入 {out_path}", file=sys.stderr)
    return _verify_emit.emit_and_exit("seed_beta_api_stress", report)


if __name__ == "__main__":
    raise SystemExit(main())
