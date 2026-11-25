"""N33.3.01 (zero training): Level-1 invariant replay for plastic growth & SessionStore capsule.

Pre-registered: plans/reference/DESIGN-N33-3-01_20261125.md (D-20261125-586).
Verifies task-agnostic signature, batch independence, neutral birth (0.0 perturbation),
Net2Net differentiation (<= 1e-6), reversible dormancy, retention-checked retire/recycle,
selective plasticity gradient mask, real SessionStore round-trip, and conservative pressure trigger.

Usage: python scripts/training/n33_3_01_growth_invariants.py
Output: reports/n33_3_01_growth_invariants.json
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from taiji.continual_growth_task import IN_DIM, NUM_CLASSES  # noqa: E402
from taiji.plastic_growth import (
    ARCH_VERSION,
    GrowthPressureController,
    PlasticGrowthNet,
)  # noqa: E402
from taiji.session_store import SessionStore  # noqa: E402

CARD = "N33.3.01"
CARD_SLUG = CARD.replace(".", "-")
SCHEMA = f"taiji-{CARD_SLUG.lower()}-v1"
DESIGN = f"DESIGN-{CARD_SLUG}_20261125.md"
CAP_CPU = 10.0
OUT = ROOT / "reports" / "n33_3_01_growth_invariants.json"


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)

    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=123)
    net.eval()
    x = torch.randn(20, IN_DIM)
    with torch.no_grad():
        full = net(x)
        singles = torch.cat([net(x[i : i + 1]) for i in range(20)], dim=0)
    inv01_diff = float((full - singles).abs().max())

    net.grow_neutral_(16, seed=456)
    with torch.no_grad():
        after_birth = net(x)
    inv02_birth_diff = float((full - after_birth).abs().max())

    net.differentiate_(2, ratio=0.35)
    with torch.no_grad():
        after_diff = net(x)
    inv02_split_diff = float((after_birth - after_diff).abs().max())

    net.mature_(list(range(8)))
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(tmpdir)
        rec = net.to_session_record("subj-inv", pending=[{"id": "t1"}])
        store.save(rec)
        restored = PlasticGrowthNet.from_session_record(store.load("subj-inv"))
    with torch.no_grad():
        inv03_store_diff = float((after_diff - restored(x)).abs().max())
        inv03_stage_exact = torch.equal(net.stage, restored.stage)

    saved = net.set_dormant_(list(range(8, net.hidden_dim)))
    net.wake_(saved)
    with torch.no_grad():
        inv_wake_diff = float((after_diff - net(x)).abs().max())

    rollback_ok = not net.retire_([0], score_fn=lambda m: float(m.active_units()), max_drop=0.5)
    with torch.no_grad():
        net.fc_out.weight[:, -1] = 0.0
    pre_recycle = net(x).detach()
    commit_ok = net.retire_([net.hidden_dim - 1], score_fn=lambda m: 1.0, max_drop=0.02)
    removed = net.recycle_()
    with torch.no_grad():
        recycle_diff = float((pre_recycle - net(x)).abs().max())

    # Selective plasticity gradient check
    net2 = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=789)
    net2.mature_()
    net2.grow_neutral_(8, seed=999)
    with torch.no_grad():
        net2.fc_out.weight[:, 8:] = 0.05
    loss = F.cross_entropy(net2(x), torch.randint(0, NUM_CLASSES, (20,)))
    loss.backward()
    net2.apply_selective_plasticity_grad_mask(freeze_mature=True)
    inv06_mature_grad = float(net2.fc_in.weight.grad[:8].abs().max())
    inv06_immature_grad = float(net2.fc_in.weight.grad[8:].abs().max())

    inv08_sig = list(inspect.signature(PlasticGrowthNet.forward).parameters)

    ctrl_quiet = GrowthPressureController(window=3, plateau_eps=0.015, error_threshold=0.15)
    quiet_events = [ctrl_quiet.observe(0.08) for _ in range(5)]
    ctrl_press = GrowthPressureController(window=3, plateau_eps=0.015, error_threshold=0.15)
    press_events = [ctrl_press.observe(e) for e in (0.40, 0.395, 0.392)]

    checks: dict[str, Any] = {
        "inv01_batch_diff": inv01_diff,
        "inv02_neutral_birth_diff": inv02_birth_diff,
        "inv02_differentiate_diff": inv02_split_diff,
        "inv03_store_roundtrip_diff": inv03_store_diff,
        "inv03_stage_exact": bool(inv03_stage_exact),
        "inv06_mature_grad_max": inv06_mature_grad,
        "inv06_immature_grad_max": inv06_immature_grad,
        "inv08_forward_params": inv08_sig,
        "wake_roundtrip_diff": inv_wake_diff,
        "retire_rollback_ok": bool(rollback_ok),
        "retire_commit_recycle_exact": bool(commit_ok and removed == 1 and recycle_diff <= 1e-6),
        "no_pressure_events": quiet_events,
        "plateau_pressure_events": press_events,
    }
    passed = (
        inv01_diff <= 1e-6
        and inv02_birth_diff <= 1e-6
        and inv02_split_diff <= 1e-6
        and inv03_store_diff == 0.0
        and inv03_stage_exact
        and inv06_mature_grad == 0.0
        and inv06_immature_grad > 0.0
        and inv08_sig == ["self", "x"]
        and inv_wake_diff == 0.0
        and rollback_ok
        and checks["retire_commit_recycle_exact"]
        and all(e == 0 for e in quiet_events)
        and press_events == [0, 0, 24]
    )
    exit_ = "invariants_ready" if passed else "invariants_failure"

    payload = {
        "schema": SCHEMA,
        "card": CARD,
        "design": DESIGN,
        "arch_version": ARCH_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"exit": exit_, "checks": checks, "cost": payload["cost"]}, ensure_ascii=False)
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
