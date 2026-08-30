from seed_platform.workbench import WorkbenchOutcome, WorkbenchStructuralEvidence


def test_structural_evidence_payload_has_stable_non_recursive_identity() -> None:
    outcome = WorkbenchOutcome(
        request_id="request-1",
        intent_id="intent-1",
        call_id="call-1",
        capability_id="workspace.read",
        snapshot_id="snapshot-1",
        status="success",
        success=True,
        result={"path": "README.md", "content": "ok"},
        tick=1,
    )
    evidence = WorkbenchStructuralEvidence.from_outcome(
        outcome,
        task_slice_id="slice-1",
        partition="train",
        usage=0.8,
        resource_pressure=0.2,
        prediction_error=0.7,
        learning_gain=0.0,
        holdout_transfer=0.0,
    )

    payload = evidence.to_payload()

    assert payload["evidence_id"] == evidence.evidence_id
    assert evidence.evidence_id.startswith("workbench-structural:")