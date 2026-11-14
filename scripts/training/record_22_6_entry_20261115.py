"""Record the 22.6 entry decision: what the gate allows and what it does not.

22.6 opens with "父身份为22.5接受的同一学习父" and 22.5.2 closes with
"check②B资格未过不进入22.6". Both are preconditions on the subject, not on the
task, and neither is satisfied here. This recorder writes that down as a named
verdict instead of letting a fixture card drift into a capability claim.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/manifests/taiji_u2_22_6_0_fixture_20261115.json"
FIXTURE = ROOT / "reports/taiji_u2_22_6_0_fixture_20261115.json"
OUT = ROOT / "reports/taiji_u2_22_6_entry_20261115.json"

GATE = [
    {
        "id": "G1",
        "requirement": "父身份为22.5接受的同一学习父，每个前后版本沿该谱系。",
        "state": "absent",
        "evidence": (
            "accepted learning parent 不存在；native_parent_qualified=false；"
            "P3.10–P3.31 无任何能力证据"
        ),
        "blocking": True,
    },
    {
        "id": "G2",
        "requirement": "check②B资格未过不进入22.6。",
        "state": "blocked",
        "evidence": "22.5 的 check② 从未按其自身口径执行；P3.4–P3.9 全为替代夹具",
        "blocking": True,
    },
    {
        "id": "G3",
        "requirement": (
            "22.5.1 B父门：全应答轮成功率≥0.80、真实反馈学习增益≥0.05、"
            "world配对95%下界>0且3seed同向、纠正率≥0.80、A保持损伤≤0.02。"
        ),
        "state": "unmeasured",
        "evidence": "五项均无读数；无父即无 A 保持对象",
        "blocking": True,
    },
    {
        "id": "G4",
        "requirement": "系数、单位、期限在22.4冻结。",
        "state": "partial",
        "evidence": (
            "deadline 已在 world contract 的 permissions.public 中；" "效用系数与归一成本单位未冻结"
        ),
        "blocking": False,
    },
    {
        "id": "G5",
        "requirement": (
            "22.6 资源：计划 8,000、余量 1,600、合计帽 9,600 CPU／19,200 wall／" "check 上限 2。"
        ),
        "state": "available",
        "evidence": "本卡 0 优化器调用、0 模型前向、0 world 步进，预算未消费",
        "blocking": False,
    },
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        print(json.dumps({"verdict": "refused", "reason": "report_already_exists"}))
        return 2
    if not FIXTURE.exists():
        print(json.dumps({"verdict": "refused", "reason": "fixture_report_missing"}))
        return 2
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen":
        print(json.dumps({"verdict": "refused", "reason": "contract_not_frozen"}))
        return 2
    if fixture.get("verdict") != "fixture_valid" or contract["parent"]["present"] is not False:
        print(json.dumps({"verdict": "refused", "reason": "entry_preconditions_not_as_expected"}))
        return 2

    start = time.perf_counter()
    pins = {
        "module": digest(ROOT / "taiji/u2_drive_task.py"),
        "manifest": digest(MANIFEST),
        "fixture_script": digest(ROOT / "scripts/training/record_22_6_0_fixture_20261115.py"),
    }
    blocking = [row["id"] for row in GATE if row["blocking"]]
    receipt = {
        "schema": "taiji-u2-22-6-entry-v1",
        "card": "22.6",
        "date": "2026-11-15",
        "verdict": "entry_limited_to_fixture_parent_absent",
        "capability_entry": False,
        "fixture_entry": True,
        "blocking_rows": blocking,
        "gate": GATE,
        "allowed_now": [
            "22.6.0: freeze the C task, the operation rights, the information rights "
            "and the resource rights",
            "fixture validity checks and their red directions",
        ],
        "forbidden_now": [
            "22.6.1 gain/risk/cost predictor",
            "22.6.2 policy versus fixed schedule, reactive control and wait control",
            "22.6.3 two dependent improvements with a fixed-parent counterfactual",
            "22.6.4 check①/check②",
            "22.6.5 whole-package closeout",
            "any capability claim of any kind",
        ],
        "why_not_forced": (
            "读卡而不读父，会把一次夹具通过记成能力通过；" "这恰是本项目反复拒绝的那种改判。"
        ),
        "fixture": {
            "report": str(FIXTURE.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": digest(FIXTURE),
            "verdict": fixture["verdict"],
            "seeds": [row["seed"] for row in fixture["seeds"]],
            "criteria_sha256": contract["criteria_sha256"],
        },
        "source_pins": pins,
        "discipline": {
            "optimizer_calls": 0,
            "model_forwards": 0,
            "world_steps": 0,
            "sealed_reads": 0,
            "capability_claims": 0,
            "checks_consumed": 0,
            "budget_consumed_cpu": 0.0,
        },
        "measured": {"wall_seconds": 0.0},
    }
    receipt["measured"]["wall_seconds"] = round(time.perf_counter() - start, 6)
    OUT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"verdict": receipt["verdict"], "blocking": blocking}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
