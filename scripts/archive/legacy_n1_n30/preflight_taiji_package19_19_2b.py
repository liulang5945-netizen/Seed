"""19.2b preflight: the 19.4 checks re-run on the 32-d span-pointer candidate.

The checks are reused verbatim from ``preflight_taiji_18_1.py`` (six cells of 3
engineering Adam, bundle-before-eval ordering, save -> restore bitwise equality
on parameters / Adam moments / teacher-forced distribution / copy state,
parent-sha and candidate-sha binding, module-drift rejection, one failure
injection).  Only three things change:

* candidates are built with ``span_bottleneck=32`` (4,194-parameter span head);
* the expected ``candidate_sha_v2`` prefix is recomputed for the new bundle
  (the format change alone never re-verifies, a parameter change must);
* report / manifest targets and the step label.

No formal training starts here: engineering steps are discarded, no confirm read.

Run: ``python scripts/training/preflight_taiji_package19_19_2b.py``
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "scripts/training/preflight_taiji_18_1.py"
REPORT = ROOT / "reports/taiji_preflight_19_2b_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_preflight_19_2b_20261104.json"
OUTPUT_DIR = ROOT / "output/taiji_preflight_19_2b"
SPAN_BOTTLENECK = 32
PARENT = ROOT / "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"


def relabel(path: Path, schema: str, step: str, package: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = schema
    payload["step"] = step
    payload["package"] = package
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def expected_prefix() -> str:
    from taiji.candidate_identity import candidate_identity
    from taiji.selection_routing_checkpoint import build_routing_candidate

    candidate, _optimizer = build_routing_candidate(
        PARENT, 28031, "A_route_lock", span_bottleneck=SPAN_BOTTLENECK
    )
    if sum(parameter.numel() for parameter in candidate.span_head.parameters()) != 4194:
        raise RuntimeError("span head is not the 4,194-parameter 19.2b shape")
    return candidate_identity(candidate)["candidate_sha_v2"][:8]


def main() -> None:
    prefix = expected_prefix()
    spec = importlib.util.spec_from_file_location("preflight181_2b", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from taiji.selection_routing_checkpoint import build_routing_candidate

    module.REPORT = REPORT
    module.MANIFEST = MANIFEST
    module.OUTPUT_DIR = OUTPUT_DIR
    module.EXPECTED_SHA_PREFIX = prefix
    module.build_routing_candidate = lambda parent, seed, arm: build_routing_candidate(
        parent, seed, arm, span_bottleneck=SPAN_BOTTLENECK
    )
    code = module.main()
    report = relabel(REPORT, "taiji-package19-preflight-v1", "N19.2b", "19.2b")
    manifest = relabel(MANIFEST, "taiji-package19-preflight-manifest-v1", "N19.2b", "19.2b")
    report["design"] = {
        "span_head": "SpanPointerHead 32-d bottleneck, 4,194 parameters",
        "expected_candidate_sha_v2_prefix": prefix,
        "unchanged": ["alpha/beta routing", "encoder", "transport operator", "gate numbers"],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "module_exit": code,
                "expected_prefix": prefix,
                "checks": report.get("checks"),
                "gates": report.get("gates"),
                "assertions": report.get("assertions"),
                "cost": report.get("cost"),
                "manifest_cost": manifest.get("cost"),
            },
            ensure_ascii=False,
            default=str,
        )[:1400]
    )


if __name__ == "__main__":
    main()
