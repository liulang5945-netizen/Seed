"""19.4 preflight: re-run the package-18.1 engineering checks on the package-19 graph.

The checks themselves are reused verbatim (six cells of 3 engineering Adam, a
single-update restore equivalence, bundle/progress/step ordering, parent-sha and
candidate-sha binding, module-drift rejection).  Only the report targets and the
step label change, plus the identity block and gate summary are refreshed.

No formal training starts here: engineering steps are discarded, no confirm read.

Run: ``python scripts/training/preflight_taiji_package19_19_4.py``
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "scripts/training/preflight_taiji_18_1.py"
REPORT = ROOT / "reports/taiji_preflight_19_4_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_preflight_19_4_20261104.json"


def relabel(path: Path, schema: str, step: str, package: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = schema
    payload["step"] = step
    payload["package"] = package
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    spec = importlib.util.spec_from_file_location("preflight181", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPORT = REPORT
    module.MANIFEST = MANIFEST
    code = module.main()
    report = relabel(REPORT, "taiji-package19-preflight-v1", "N19.4", "19")
    manifest = relabel(MANIFEST, "taiji-package19-preflight-manifest-v1", "N19.4", "19")
    print(
        json.dumps(
            {
                "module_exit": code,
                "checks": report.get("checks"),
                "gates": report.get("gates"),
                "cost": report.get("cost"),
                "manifest_cost": manifest.get("cost"),
            },
            ensure_ascii=False,
            default=str,
        )[:1200]
    )


if __name__ == "__main__":
    main()
