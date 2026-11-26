"""19.3 pricing: re-measure the six-cell cost on the current (cursor + UTF-8) graph.

The component measurements and the extrapolation formula are reused verbatim from
the 18.0 pricing script so that 18.0 and 19.3 prices stay comparable: only the
report/manifest targets and the step label change, and the semantic identity is
recomputed because the graph gained copy_cursor / copy_record / prefix_cursor.

No formal training starts here: the four optimiser steps are discarded, the
temporary bundle is removed, confirm is not read.

Run: ``python scripts/training/price_taiji_package19_19_3.py``
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "scripts/training/price_taiji_data_premise_18_0.py"
REPORT = ROOT / "reports/taiji_pricing_19_3_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_pricing_19_3_20261104.json"
PRICE_CPU_CAP = 30000.0


def load_pricing_module():
    spec = importlib.util.spec_from_file_location("price18", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_pricing_module()
    module.REPORT = REPORT
    module.MANIFEST = MANIFEST
    module.main()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    compare = module.compare_module()
    candidate, _optimizer = module.build_routing_candidate(module.PARENT, module.SEED, module.ARM)
    identity = compare.candidate_identity(candidate)
    report["schema"] = "taiji-package19-pricing-v1"
    report["step"] = "N19.3"
    report["package"] = "19"
    report["graph"] = "copy_cursor + copy_record + prefix_cursor + utf8_mask"
    report["identity"] = identity
    report["caps"] = {"max_adam": 0, "max_cpu": 200.0, "watchline_cpu": PRICE_CPU_CAP}
    report["gates"] = {
        "optimizer_steps": 0,
        "training_updates": 0,
        "confirm_reads": 0,
        "six_cells_protected_le_cap": bool(
            report["budget"]["six_cells_with_contingency_b_protected"] <= PRICE_CPU_CAP
        ),
        "candidate_identity_recorded": bool(identity),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    MANIFEST.write_text(
        json.dumps(
            {
                "schema": "taiji-package19-pricing-manifest-v1",
                "step": "N19.3",
                "date": "2026-11-04",
                "generated_by": str(MODULE_PATH.relative_to(ROOT)),
                "wrapper_sha256": __import__("hashlib")
                .sha256(Path(__file__).read_bytes())
                .hexdigest(),
                "report": {
                    "path": str(REPORT.relative_to(ROOT)),
                    "sha256": __import__("hashlib").sha256(REPORT.read_bytes()).hexdigest(),
                    "bytes": REPORT.stat().st_size,
                },
                "identity": identity,
                "budget": {
                    key: report["budget"][key]
                    for key in (
                        "six_cells_bare",
                        "six_cells_protected",
                        "six_cells_with_contingency_b_protected",
                    )
                },
                "cost": report["cost"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "six_cells_protected": round(
                    report["budget"]["six_cells_with_contingency_b_protected"], 1
                ),
                "watchline_cpu": PRICE_CPU_CAP,
                "identity": identity,
                "gates": report["gates"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
