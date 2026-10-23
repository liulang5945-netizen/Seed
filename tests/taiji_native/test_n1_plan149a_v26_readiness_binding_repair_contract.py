from __future__ import annotations

from pathlib import Path

from scripts.training import run_taiji_n1_v26_abstention_length_after_n148_repair as repair


def test_repair_binds_nested_verifier_and_adds_readiness_source_hashes() -> None:
    n148_builder = repair.n148_builder
    original_source_hashes = repair.frozen_eval._source_hashes
    marker = object()
    original_alias = getattr(n148_builder, "n144_builder", marker)
    before_builder = repair.sha256(repair.face_builder.__file__)
    before_evaluator = repair.sha256(repair.frozen_eval.__file__)
    try:
        bindings = repair.configure_runtime()
        hashes = repair.frozen_eval._source_hashes({"source_hashes": {}}, [])

        assert n148_builder.n144_builder is n148_builder.n147_builder.n144_builder
        assert bindings["verifier"] is n148_builder.n144_builder._verify_hashes
        assert hashes[str(repair.REPAIR_PLAN.resolve())] == repair.sha256(repair.REPAIR_PLAN)
        assert hashes[str(Path(repair.__file__).resolve())] == repair.sha256(Path(repair.__file__))
        assert hashes[str(repair.TEST.resolve())] == repair.sha256(repair.TEST)
        assert hashes[str(repair.FAILURE_RECEIPT.resolve())] == repair.sha256(
            repair.FAILURE_RECEIPT
        )
        assert repair.sha256(repair.face_builder.__file__) == before_builder
        assert repair.sha256(repair.frozen_eval.__file__) == before_evaluator
    finally:
        repair.frozen_eval._source_hashes = original_source_hashes
        if original_alias is marker:
            delattr(n148_builder, "n144_builder")
        else:
            n148_builder.n144_builder = original_alias
        repair._CONFIGURED = False
        repair._ORIGINAL_SOURCE_HASHES = None


def test_repair_configuration_does_not_restore_or_query_models() -> None:
    n148_builder = repair.n148_builder
    outputs = (
        repair.frozen_eval.READINESS,
        repair.frozen_eval.READINESS_ATTEMPT,
        repair.frozen_eval.SOURCE_BINDING,
        repair.frozen_eval.ATTEMPT,
        repair.frozen_eval.PROGRESS,
        repair.frozen_eval.RESULT,
    )
    existed = {path: path.exists() for path in outputs}
    original_source_hashes = repair.frozen_eval._source_hashes
    marker = object()
    original_alias = getattr(n148_builder, "n144_builder", marker)
    try:
        repair.configure_runtime()

        # 生命周期门控：原断言「六份产物都必须不存在」是 pre-run 世界假设；10-19 的
        # readiness/诊断阶段已跑完，回执已随 reports/ 入库（jsonl 为未入库产物），
        # 「不存在」在任何 checkout 上都不再成立。改为不随生命周期变化的契约：
        # 只做内存绑定的 configure_runtime() 不得创建或删除任何阶段产物（前=后）。
        for path in outputs:
            assert path.is_relative_to(repair.ROOT)
        assert {path: path.exists() for path in outputs} == existed
    finally:
        repair.frozen_eval._source_hashes = original_source_hashes
        if original_alias is marker:
            delattr(n148_builder, "n144_builder")
        else:
            n148_builder.n144_builder = original_alias
        repair._CONFIGURED = False
        repair._ORIGINAL_SOURCE_HASHES = None
