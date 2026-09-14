"""Directed tests for the runtime-owned K/G attachment contract.

Covers the frozen fail-closed load order of
``api/taiji_runtime_attachment.py``: manifest self-digest, lineage,
per-artifact digests, embedded invariants, tamper and wrong-cell-mixing
rejection, plus one opt-in attach/detach cycle on a real ``SeedRuntime``.
No training happens; the artifacts on disk are only ever read.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from api import taiji_runtime_attachment as attachment
from api.seed_runtime import SeedRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_default_runtime_rollout_attachment_v1.json"
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "attachment_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def _artifact_paths(manifest: dict) -> tuple[Path, ...]:
    paths: list[Path] = []
    for cell in manifest.get("cells", ()):
        for artifact in (cell.get("artifacts") or {}).values():
            raw = artifact.get("path") if isinstance(artifact, dict) else None
            if raw:
                paths.append(Path(str(raw)))
    return tuple(paths)


def _require_local_artifacts(manifest: dict) -> None:
    """Skip when the manifest's on-disk training artifacts are absent.

    The manifest ships in-repo, but the P4.14 artifacts it points at are local
    training products under ``output/`` that are deliberately not committed, so
    CI has none.  Tests that must read real payloads skip rather than fail there
    (same intent as ci.yml: "Legacy real-checkpoint tests skip when local
    artifacts are absent").
    """

    missing = [path for path in _artifact_paths(manifest) if not path.exists()]
    if missing:
        pytest.skip(
            f"{len(missing)} local P4.14 artifact(s) absent (e.g. {missing[0]}); "
            "this test reads real on-disk artifacts that are not committed"
        )


def test_module_does_not_import_research_scripts() -> None:
    source = Path(attachment.__file__).read_text(encoding="utf-8")
    assert "from scripts" not in source
    assert "import scripts" not in source


def test_manifest_self_digest_rejects_tampering(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["consumption_contract"]["forbidden"] = ["nothing"]
    with pytest.raises(attachment.AttachmentRefused, match="self-digest"):
        attachment._load_manifest(_write_manifest(tmp_path, manifest))


def test_artifact_digest_mismatch_rejected(tmp_path: Path) -> None:
    manifest = _manifest()
    _require_local_artifacts(manifest)
    manifest["cells"][0]["artifacts"]["g"]["digest"] = "0" * 64
    with pytest.raises(attachment.AttachmentRefused, match="digest mismatch"):
        attachment._load_cell_payloads(manifest, 0)


def test_missing_artifact_rejected(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["cells"][0]["artifacts"]["k1"]["path"] = str(tmp_path / "missing_k1.pt")
    with pytest.raises(attachment.AttachmentRefused, match="missing"):
        attachment._load_cell_payloads(manifest, 0)


def test_wrong_cell_mixing_rejected() -> None:
    manifest = _manifest()
    _require_local_artifacts(manifest)
    assert attachment._mixing_probe_rejected(manifest) is True


def test_tamper_probes_rejected() -> None:
    manifest = _manifest()
    _require_local_artifacts(manifest)
    payloads = attachment._load_cell_payloads(manifest, 0)["payloads"]
    results = attachment._tamper_probes_rejected(payloads)
    assert all(results.values())


def test_invariants_pinned_from_checkpoints() -> None:
    manifest = _manifest()
    _require_local_artifacts(manifest)
    payloads = attachment._load_cell_payloads(manifest, 0)["payloads"]
    checks = attachment._invariant_checks(payloads)
    assert all(checks.values())
    assert checks["k1_fact_threshold"] and checks["k2_fact_threshold"]
    semantic, transition, g_learner = attachment._build_learners(payloads)
    assert int(semantic.parameter_count + transition.parameter_count) == 5648
    assert int(g_learner.parameter_count) == 17


def test_lineage_binds_p4_14_and_p4_1() -> None:
    manifest = _manifest()
    lineage = attachment._verify_lineage(manifest)
    assert lineage["p4_14_outcome"] == "joint_course_supported"
    assert lineage["p4_14_passing_cells"] == 4
    assert len(lineage["g_parent_manifest_digest_expected"]) == 64


def test_verify_attachment_cli_rejects_corrupt_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["cells"][1]["artifacts"]["k2"]["digest"] = "1" * 64
    corrupt = _write_manifest(tmp_path, manifest)
    result = attachment.main(["--verify-attachment", str(corrupt), "1"])
    assert result == 1


@pytest.mark.parametrize("cell_index", [0, 1, 2, 3])
def test_full_preflight_passes_for_every_cell(cell_index: int) -> None:
    _require_local_artifacts(_manifest())
    result = attachment.verify_attachment(MANIFEST_PATH, cell_index)
    assert result["passed"] is True
    assert all(result["invariant_checks"].values())
    assert all(result["learner_checks"].values())
    assert result["wrong_cell_mixing_rejected"] is True


def test_seed_runtime_attach_detach_cycle() -> None:
    _require_local_artifacts(_manifest())
    runtime = SeedRuntime.load()
    assert runtime.k_g_attachment_status()["attached"] is False
    status = runtime.attach_k_g_state(MANIFEST_PATH, 0)
    assert status["attached"] is True
    assert status["cell_index"] == 0
    assert status["fit_called"] is False
    assert runtime.status()["k_g_attachment"]["attached"] is True
    detached = runtime.detach_k_g_state()
    assert detached["detached"] is True
    assert runtime.k_g_attachment_status()["attached"] is False


def test_seed_runtime_records_refusal_and_stays_unattached(tmp_path: Path) -> None:
    runtime = SeedRuntime.load()
    manifest = copy.deepcopy(_manifest())
    manifest["cells"][0]["artifacts"]["g"]["digest"] = "2" * 64
    corrupt = _write_manifest(tmp_path, manifest)
    with pytest.raises(attachment.AttachmentRefused):
        runtime.attach_k_g_state(corrupt, 0)
    status = runtime.k_g_attachment_status()
    assert status["attached"] is False
    assert len(status["recent_refusals"]) == 1
    assert "digest mismatch" in status["recent_refusals"][0]["reason"]
