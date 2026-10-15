"""Genuine frozen lineage contracts, explicitly local-only by repository policy."""

from copy import deepcopy

from scripts.training import audit_taiji_n1_group_relative_paired as audit
from scripts.training import audit_taiji_n1_local_relation as engineering

LOCAL_ONLY_ARTIFACTS = (
    "data/checkpoints/n1_contextual_question_train_20261015/final.pt",
    "data/checkpoints/n1_local_question_train_20261015/final.pt",
    "data/checkpoints/n1_relation_paired_train_20261015/s13_position_zero.pt",
)


def test_genuine_frozen_lineage_and_seed11_engineering_identity():
    audit.torch.set_num_threads(1)
    audit.readiness()
    transfer = audit.runner.transfer
    _, payload, _ = transfer.source_payload()
    model, optimizer = transfer.migrate(payload, 11)
    ready = transfer.json.loads(engineering.REPORT.read_text(encoding="utf-8"))
    expected = deepcopy(
        transfer.torch.load(ready["restore_check"]["path"], map_location="cpu", weights_only=True)[
            "candidate"
        ]
    )
    for session in engineering.SESSIONS:
        del expected["sessions"][session]
    assert transfer.source.paired.fingerprint(
        model.bundle(optimizer)
    ) == transfer.source.paired.fingerprint(expected)


def test_genuine_contextual_parent_migration():
    from scripts.training import _taiji_n1_contextual_question_transfer as transfer

    _, payload, _ = transfer.source_payload()
    model, optimizer = transfer.migrate(payload)
    assert len(optimizer.state) == len(list(model.parameters())) - len(transfer.NEW_PARAMETERS)
