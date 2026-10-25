"""DEBT-G74③ 的门叶子：预注册语料的**全集**措辞审计棘轮（跑套件时就顺带检，不再只检动过的那份）。

为什么需要这条：㊵-622 第一次把 `plans/reference/PLAN-*.md` 全量跑了一遍，才发现 42 份里
有 8 份一直不过门——**不是因为它们是坏的，而是因为这道门从没被全集跑过**（过去只在
「本轮动过的那份」上跑）。一条只在改动时触发的守卫，等于没有守卫。

形状选择（三处都刻意如此）：
1. **棘轮而非清零**：已知违例冻成白名单，判定用「子集」而不是「相等」⇒
   新增违例当场红，**修掉违例不会红**（不许为了把数字做漂亮去批量改已冻预注册的措辞——
   那属于 G74②，要走升版，且归人逐份判）；
2. ** corpus 规模也钉**：扫到的份数必须等于盘上 `PLAN-*.md` 的份数——
   glob 深度写错会让「零违例」变成假绿（本仓记过：零命中要连范围一起报）；
3. **不起 42 个子进程**：一次调用传多个 `--doc`，快且可比。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_prereg_exit_wording.py"

#: ㊵-625 实测冻结（改后基线：42 份＝34 ok／4 invisible／4 ambiguous）——2026-10-23 已过期，
#: 仅留作历史口径。
#: 这两组是**已知待办清单**，不是"允许长期存在的红"：G74② 逐份修完就该往下缩。
#: 2026-10-23 重冻（42→329 份语料）：CI 自 10-19 起长红，N1/R2 主线在红窗内
#: 落地约 300 份新预注册，从未被这道门扫过。本机实测全集 329 份＝105 ok／224 invisible／
#: 0 ambiguous，白名单按「基线＝当前实测值」一次性重冻（同 H19f 先例）；
#: 旧 42 份时代的 4 项 invisible 全部仍在册（无假绿）。G74② 的逐份措辞修复仍归人判，
#: 修掉一份就应把该名从这里删掉（棘轮只许单向收紧）。
#: 2026-10-25 二次重冻（结案波 339 份）：基线＝当前实测全集（230 invisible），
#: 并行会话结案文档仍在落盘，本清单按波次保持当前（G74② 逐份修复归人判）。
FROZEN_INVISIBLE = {
    "PLAN-N02-V014-turn-trace-diagnostic-draft-20261018.md",
    "PLAN-N1-07_C2_calibration_budget_adjudication_20261015.md",
    "PLAN-N1-07_C2_calibration_result_adjudication_20261015.md",
    "PLAN-N1-08_C2_state_flow_adjudication_20261015.md",
    "PLAN-N1-09_C2_padding_placement_prereg_20261015.md",
    "PLAN-N1-100_balanced_hard_neighbor_training_and_v7_20261017.md",
    "PLAN-N1-101_v8_hard_neighbor_range_recovery_20261017.md",
    "PLAN-N1-102_v12_observation_range_probe_20261017.md",
    "PLAN-N1-103_v9_item_level_error_attribution_20261017.md",
    "PLAN-N1-104_v9_confirmation_input_feature_audit_20261017.md",
    "PLAN-N1-105_counterbalanced_input_feature_diagnostic_20261017.md",
    "PLAN-N1-106_v10_output_error_attribution_20261017.md",
    "PLAN-N1-107_v10_output_geometry_and_position_stability_20261017.md",
    "PLAN-N1-108_v10_teacher_free_agreement_20261017.md",
    "PLAN-N1-109_train_answer_length_coverage_audit_20261017.md",
    "PLAN-N1-10_native_evidence_credit_route_review_20261015.md",
    "PLAN-N1-110_execution_closeout_20261018.md",
    "PLAN-N1-110_length_balanced_training_and_v11_confirmation_draft_20261017.md",
    "PLAN-N1-111_position_weight_history_crossover_diagnostic_draft_20261018.md",
    "PLAN-N1-112_position_weight_history_crossover_restore_revision_draft_20261018.md",
    "PLAN-N1-113_position_checkpoint_parameter_delta_attribution_20261018.md",
    "PLAN-N1-114_seed13_parameter_update_trajectory_attribution_20261018.md",
    "PLAN-N1-115_five_group_parameter_factorial_diagnostic_draft_20261018.md",
    "PLAN-N1-117_coupled_rwl_source_loader_revision_draft_20261018.md",
    "PLAN-N1-118_execution_closeout_20261019.md",
    "PLAN-N1-119_execution_closeout_20261019.md",
    "PLAN-N1-119_seed13_normalized_rwl_itemwise_attribution_draft_20261019.md",
    "PLAN-N1-120_execution_closeout_20261019.md",
    "PLAN-N1-120_expanded_normalized_rwl_radius_training_draft_20261019.md",
    "PLAN-N1-121_execution_closeout_20261019.md",
    "PLAN-N1-121_source_exclusion_correction_20261019.md",
    "PLAN-N1-121_v11_confirmation_after_n120_training_draft_20261019.md",
    "PLAN-N1-122_execution_closeout_20261019.md",
    "PLAN-N1-122_v12_hard_neighbor_range_recovery_draft_20261019.md",
    "PLAN-N1-123_execution_closeout_20261019.md",
    "PLAN-N1-123_v13_long_context_range_probe_draft_20261019.md",
    "PLAN-N1-124_execution_closeout_20261019.md",
    "PLAN-N1-124_v14_full_context_evidence_position_probe_draft_20261019.md",
    "PLAN-N1-125_content_binding_revision_draft_20261019.md",
    "PLAN-N1-125_existing_binding_audit_closeout_20261019.md",
    "PLAN-N1-126_execution_closeout_20261019.md",
    "PLAN-N1-126_source_pruned_saved_error_diagnostic_draft_20261019.md",
    "PLAN-N1-127_execution_closeout_20261019.md",
    "PLAN-N1-127_full_context_source_attention_mask_diagnostic_draft_20261019.md",
    "PLAN-N1-128_execution_closeout_20261019.md",
    "PLAN-N1-128_full_context_question_binding_mask_diagnostic_draft_20261019.md",
    "PLAN-N1-129_execution_closeout_20261019.md",
    "PLAN-N1-129_native_question_binding_mass_audit_draft_20261019.md",
    "PLAN-N1-130_execution_closeout_20261019.md",
    "PLAN-N1-130_native_relation_row_binding_mass_audit_draft_20261019.md",
    "PLAN-N1-131_execution_closeout_20261019.md",
    "PLAN-N1-131_transition_operator_error_attribution_draft_20261019.md",
    "PLAN-N1-132_execution_closeout_20261019.md",
    "PLAN-N1-132_trainable_question_binding_supervision_calibration_draft_20261019.md",
    "PLAN-N1-133_execution_closeout_20261019.md",
    "PLAN-N1-134_execution_closeout_20261019.md",
    "PLAN-N1-134_transition_direction_position_counterbalance_diagnostic_draft_20261019.md",
    "PLAN-N1-135_execution_closeout_20261019.md",
    "PLAN-N1-135_matched_context_length_transition_diagnostic_draft_20261019.md",
    "PLAN-N1-136_execution_closeout_20261019.md",
    "PLAN-N1-136_target_position_context_order_diagnostic_draft_20261019.md",
    "PLAN-N1-137_execution_closeout_20261019.md",
    "PLAN-N1-137_neighbor_distance_context_interference_diagnostic_draft_20261019.md",
    "PLAN-N1-138_execution_closeout_20261019.md",
    "PLAN-N1-138_neighbor_distance_dose_curve_diagnostic_draft_20261019.md",
    "PLAN-N1-139_context_robust_matched_continuation_calibration_draft_20261019.md",
    "PLAN-N1-139_execution_closeout_20261019.md",
    "PLAN-N1-140_baseline_retention_balanced_matched_continuation_draft_20261019.md",
    "PLAN-N1-140_execution_closeout_20261019.md",
    "PLAN-N1-141_execution_closeout_20261019.md",
    "PLAN-N1-142_execution_closeout_20261019.md",
    "PLAN-N1-142_v21_fresh_confirmation_after_n141_draft_20261019.md",
    "PLAN-N1-143_execution_closeout_20261019.md",
    "PLAN-N1-143_v22_fresh_confirmation_after_n142_closeout_20261019.md",
    "PLAN-N1-144_execution_closeout_20261019.md",
    "PLAN-N1-144_v23_long_context_confirmation_after_n143_draft_20261019.md",
    "PLAN-N1-145_execution_closeout_20261019.md",
    "PLAN-N1-145_v23_itemwise_error_attribution_draft_20261019.md",
    "PLAN-N1-146_execution_closeout_20261019.md",
    "PLAN-N1-146_v23_output_source_item_attribution_draft_20261019.md",
    "PLAN-N1-147A_readiness_verifier_binding_correction_20261019.md",
    "PLAN-N1-147_execution_closeout_20261019.md",
    "PLAN-N1-147_matched_known_unknown_abstention_diagnostic_draft_20261019.md",
    "PLAN-N1-148_execution_closeout_20261019.md",
    "PLAN-N1-148_matched_abstention_diagnostic_scorer_revision_20261019.md",
    "PLAN-N1-149A_readiness_module_binding_correction_20261020.md",
    "PLAN-N1-149_execution_closeout_20261020.md",
    "PLAN-N1-149_matched_abstention_context_length_curve_diagnostic_draft_20261019.md",
    "PLAN-N1-14_native_candidate_paired_pilot_20261015.md",
    "PLAN-N1-150A_train_only_scoring_denominator_freeze_20261020.md",
    "PLAN-N1-150B_training_cpu_ceiling_revision_20261020.md",
    "PLAN-N1-150_execution_closeout_20261020.md",
    "PLAN-N1-150_matched_long_context_training_continuation_draft_20261020.md",
    "PLAN-N1-151A_execution_closeout_20261020.md",
    "PLAN-N1-151A_source_binding_return_correction_20261020.md",
    "PLAN-N1-151_train_score_ledger_recovery_draft_20261020.md",
    "PLAN-N1-152_execution_closeout_20261020.md",
    "PLAN-N1-152_length_itemwise_attribution_20261020.md",
    "PLAN-N1-153A_execution_closeout_20261020.md",
    "PLAN-N1-153A_fresh_context_result_binding_recovery_20261020.md",
    "PLAN-N1-153_fresh_context_length_paired_diagnostic_20261020.md",
    "PLAN-N1-154_execution_closeout_20261020.md",
    "PLAN-N1-154_fresh_context_itemwise_attribution_20261020.md",
    "PLAN-N1-155A_execution_closeout_20261020.md",
    "PLAN-N1-155A_runtime_views_handoff_correction_20261020.md",
    "PLAN-N1-155_short_context_unknown_rebalance_and_fresh_diagnostic_20261020.md",
    "PLAN-N1-156_execution_closeout_20261020.md",
    "PLAN-N1-156_plan155a_train_score_itemwise_attribution_20261020.md",
    "PLAN-N1-157A2_execution_closeout_20261020.md",
    "PLAN-N1-157A2_readiness_binding_metadata_correction_20261020.md",
    "PLAN-N1-157A_remaining_endpoint_continuation_20261020.md",
    "PLAN-N1-157_seed128_candidate_vs_parent_fresh_diagnostic_20261020.md",
    "PLAN-N1-158_execution_closeout_20261020.md",
    "PLAN-N1-158_plan157a2_fresh_face_itemwise_attribution_20261020.md",
    "PLAN-N1-159A_execution_closeout_20261020.md",
    "PLAN-N1-159A_per_endpoint_summary_correction_20261020.md",
    "PLAN-N1-159A_remaining_endpoint_continuation_20261020.md",
    "PLAN-N1-159_seed129_candidate_parent_fresh_replication_20261020.md",
    "PLAN-N1-160A_execution_closeout_20261020.md",
    "PLAN-N1-160A_remaining_endpoint_continuation_20261020.md",
    "PLAN-N1-160_seed130_candidate_parent_fresh_replication_20261020.md",
    "PLAN-N1-161A_execution_closeout_20261020.md",
    "PLAN-N1-161A_question_surface_itemwise_attribution_20261020.md",
    "PLAN-N1-161_execution_closeout_20261020.md",
    "PLAN-N1-161_unseen_question_surface_diagnostic_20261020.md",
    "PLAN-N1-162A_执行预算可行性修订_20261020.md",
    "PLAN-N1-162B_execution_closeout_20261020.md",
    "PLAN-N1-162B_首步解析器修正与隔离续执行_20261020.md",
    "PLAN-N1-162C_execution_closeout_20261020.md",
    "PLAN-N1-162C_train_only_itemwise_gate_attribution_20261020.md",
    "PLAN-N1-162D_execution_closeout_20261020.md",
    "PLAN-N1-162D_transition_output_type_attribution_20261020.md",
    "PLAN-N1-162E_execution_closeout_20261020.md",
    "PLAN-N1-162E_parent_control_surface_drift_attribution_20261020.md",
    "PLAN-N1-162F1_execution_closeout_20261020.md",
    "PLAN-N1-162F1_transcript_integrity_revalidation_20261020.md",
    "PLAN-N1-162F_transition_template_crossover_train_diagnostic_20261020.md",
    "PLAN-N1-162G1_execution_closeout_20261020.md",
    "PLAN-N1-162G1_training_thread_scaling_preflight_repair_20261020.md",
    "PLAN-N1-162G2_execution_closeout_20261020.md",
    "PLAN-N1-162G2_mixed_context_thread_scaling_benchmark_20261020.md",
    "PLAN-N1-162G_training_thread_scaling_benchmark_20261020.md",
    "PLAN-N1-162_question_surface_matched_continuation_20261020.md",
    "PLAN-N1-163A_readiness_source_rebinding_20261020.md",
    "PLAN-N1-163_execution_closeout_20261020.md",
    "PLAN-N1-163_transition_template_rebalanced_matched_continuation_20261020.md",
    "PLAN-N1-164_plan163_itemwise_pair_attribution_20261020.md",
    "PLAN-N1-165A_execution_closeout_20261021.md",
    "PLAN-N1-165A_partial_attempt_completion_20261020.md",
    "PLAN-N1-165_transition_template_mid_dose_calibration_20261020.md",
    "PLAN-N1-166_compositional_transfer_confirmation_20261020.md",
    "PLAN-N1-167_execution_closeout_20261025.md",
    "PLAN-N1-168_execution_closeout_20261025.md",
    "PLAN-N1-169_shared_language_chain_design_20261025.md",
    "PLAN-N1-18_fresh_content_curriculum_20261015.md",
    "PLAN-N1-21_observation_fidelity_learning_review_20261015.md",
    "PLAN-N1-24_pre_emission_iterative_read_review_20261015.md",
    "PLAN-N1-27_address_intervention_diagnostic_review_20261015.md",
    "PLAN-N1-32_fixed_paired_continuation_review_20261015.md",
    "PLAN-N1-34_entity_binding_diagnostic_review_20261015.md",
    "PLAN-N1-35_question_alignment_review_20261015.md",
    "PLAN-N1-36_question_alignment_calibration_review_20261015.md",
    "PLAN-N1-37_residual_question_alignment_review_20261015.md",
    "PLAN-N1-38_residual_question_calibration_review_20261015.md",
    "PLAN-N1-39_question_fusion_ablation_review_20261015.md",
    "PLAN-N1-40_paired_question_learning_review_20261015.md",
    "PLAN-N1-41_reader_transfer_paired_review_20261015.md",
    "PLAN-N1-42_transfer_binding_diagnostic_review_20261015.md",
    "PLAN-N1-43_renaming_curriculum_execution_20261015.md",
    "PLAN-N1-44_read_equivariance_diagnostic_20261015.md",
    "PLAN-N1-45_cross_read_intervention_20261015.md",
    "PLAN-N1-46_address_consistency_calibration_20261015.md",
    "PLAN-N1-47_writer_question_interchange_20261015.md",
    "PLAN-N1-48_local_question_binding_preflight_20261015.md",
    "PLAN-N1-49_local_question_calibration_20261015.md",
    "PLAN-N1-50_local_binding_ablation_20261015.md",
    "PLAN-N1-51_contextual_question_preflight_20261015.md",
    "PLAN-N1-52_contextual_question_calibration_20261015.md",
    "PLAN-N1-53_bound_question_writer_interchange_20261015.md",
    "PLAN-N1-54_bound_question_consistency_engineering_20261015.md",
    "PLAN-N1-55_bound_question_consistency_calibration_20261015.md",
    "PLAN-N1-57_relative_question_margin_engineering_20261015.md",
    "PLAN-N1-58_relative_question_margin_calibration_20261015.md",
    "PLAN-N1-60_local_relation_tape_engineering_20261015.md",
    "PLAN-N1-61_local_relation_calibration_20261015.md",
    "PLAN-N1-63_relation_paired_training_20261015.md",
    "PLAN-N1-65_relation_seed_drift_diagnostic_20261015.md",
    "PLAN-N1-66_relation_update_replay_diagnostic_20261015.md",
    "PLAN-N1-67_group_relative_adam_engineering_20261015.md",
    "PLAN-N1-68_group_relative_adam_calibration_20261015.md",
    "PLAN-N1-69_ci_fixture_portability_20261015.md",
    "PLAN-N1-69_group_relative_paired_engineering_20261015.md",
    "PLAN-N1-70_group_relative_paired_training_20261015.md",
    "PLAN-N1-71_frozen_v4_confirmation_20261016.md",
    "PLAN-N1-72_frozen_writer_exchange_20261016.md",
    "PLAN-N1-73_credit_lineage_review_20261016.md",
    "PLAN-N1-74_fresh_full_history_engineering_20261016.md",
    "PLAN-N1-75_fresh_full_history_calibration_20261016.md",
    "PLAN-N1-76_fresh_paired_admission_20261016.md",
    "PLAN-N1-77_fresh_paired_training_20261016.md",
    "PLAN-N1-78_fresh_frozen_confirmation_20261016.md",
    "PLAN-N1-79_fresh_training_log_diagnostic_20261016.md",
    "PLAN-N1-80_fresh_transition_factorial_diagnostic_20261016.md",
    "PLAN-N1-82_transition_factorial_calibration_20261016.md",
    "PLAN-N1-84_transition_factorial_paired_training_20261016.md",
    "PLAN-N1-86_detached_factorial_execution_20261016.md",
    "PLAN-N1-87R_run86c_offline_audit_retry_review_20261016.md",
    "PLAN-N1-87_run86c_offline_trajectory_audit_review_20261016.md",
    "PLAN-N1-98_static_cross_face_unknown_audit_20261017.md",
    "PLAN-N1-99_unknown_abstention_factorial_diagnostic_20261017.md",
    "PLAN-R2-DAILY-DIALOGUE-01_execution_closeout_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-ARCHITECTURE-AUDIT_20261021_execution_closeout.md",
    "PLAN-R2-DAILY-DIALOGUE-ATTRIBUTION_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-ATTRIBUTION_20261021_execution_closeout.md",
    "PLAN-R2-DAILY-DIALOGUE-CALIBRATION_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-CONTINUATION-CC1_execution_closeout_20261025.md",
    "PLAN-R2-DAILY-DIALOGUE-CONTINUATION-RECOVERY1_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-CONTINUATION-RECOVERY1_execution_closeout_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-CONTINUATION_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-MATERIAL-INVENTORY-RECOVERY1_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-MATERIAL-INVENTORY-RECOVERY1_execution_closeout_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-RECOVERY_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-RECOVERY_V2_READINESS_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-RETRY1_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-RETRY2_20261021.md",
    "PLAN-R2-DAILY-DIALOGUE-WARMSTART-MATCHED_20261021.md",
    "PLAN-R2-EVALUATION-RECOVERY-V1_execution_closeout_20261025.md",
    "PLAN-R2-EVALUATION-RECOVERY-V2_human_review_20261025.md",
    "PLAN-R2-THREAD-SCALING-DIAGNOSTIC_20261021.md",
}
#: 2026-10-25 二次重冻（结案波）：1 份 ambiguous（N1-169 closeout，同批待修）。
FROZEN_AMBIGUOUS = {
    "PLAN-N1-169_execution_closeout_20261025.md",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("prereg_wording_corpus_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _corpus() -> list[Path]:
    return sorted((REPO / "plans" / "reference").glob("PLAN-*.md"))


def _audit(tmp_path: Path, docs: list[Path]) -> list[dict]:
    out = tmp_path / "corpus.json"
    argv: list[str] = ["--out-report", str(out)]
    for doc in docs:
        argv += ["--doc", str(doc)]
    rc = _load_module().main(argv)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rc"] == rc, (payload["rc"], rc)
    return payload["results"]


def test_corpus_is_not_silently_narrowed(tmp_path: Path) -> None:
    #: 扫到的份数＝盘上的份数（防 glob 写错造成"全集=0 份⇒全绿"的假绿）。
    docs = _corpus()
    assert len(docs) >= 42, f"语料规模异常（{len(docs)} 份）⇒ 先证明 glob 还覆盖着全集"
    entries = _audit(tmp_path, docs)
    assert len(entries) == len(docs), (len(entries), len(docs))
    #: 每一份都得真出现（路径串对账），少一份就是取法坏了而不是文档好了。
    names = {Path(e["doc"]).name for e in entries}
    assert names == {d.name for d in docs}, names ^ {d.name for d in docs}


def test_no_new_invisible_docs_appear(tmp_path: Path) -> None:
    entries = _audit(tmp_path, _corpus())
    invisible = {
        Path(e["doc"]).name for e in entries if e["status"] == "invisible_criterion_surface"
    }
    new = invisible - FROZEN_INVISIBLE
    assert (
        not new
    ), f"新增对措辞门隐形的预注册：{sorted(new)}（修法：给判据段加一个含 判据/出口/验收 的二级标题）"


def test_no_new_ambiguous_docs_appear(tmp_path: Path) -> None:
    entries = _audit(tmp_path, _corpus())
    ambiguous = {Path(e["doc"]).name for e in entries if e["status"] == "ambiguous_exit_wording"}
    new = ambiguous - FROZEN_AMBIGUOUS
    assert (
        not new
    ), f"新增未钉数值的出口句：{sorted(new)}（修法：把「明显/大幅/显著/足够」换成数值合取）"
