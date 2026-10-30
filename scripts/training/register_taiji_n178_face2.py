"""登记 N178 的固定新面（只对照 train 文本做泄漏检查；query/holdout 不参与建面）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.joint_state_consumption_course import build_train_episodes  # noqa: E402
from taiji.joint_state_consumption_face2 import face2_contract, leakage_check  # noqa: E402

REPORT = ROOT / "reports/taiji_n178_face2_contract_20261030.json"


def main() -> int:
    if REPORT.exists():
        raise ValueError("face2 contract is write-once")
    existing = {
        bytes(turn.frame.payload).decode("utf-8", "replace")
        for episode in build_train_episodes()
        for turn in episode.turns
    }
    payload = face2_contract()
    payload["leakage_vs_train"] = leakage_check(existing)
    payload["leakage_scope"] = "train 文本；query/holdout 未参与建面"
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "items"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
