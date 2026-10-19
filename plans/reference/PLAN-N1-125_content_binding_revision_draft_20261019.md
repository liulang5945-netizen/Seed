# PLAN-N1-125 — existing binding and saved-error audit

**Status: complete; the proposed new binding mechanism is withdrawn.** A static source audit found that the active candidate already contains question-to-observation local byte matching, observation-conditioned question context, and a question-conditioned key/value residual. Adding another generic question-binding module would duplicate existing machinery before identifying the remaining failure.

## Existing mechanism map

The current stack already matches five-byte question windows against observation windows in `LocalQuestionEvidenceCandidate.bind_question`, uses those alignments and confidence in `ContextualQuestionEvidenceCandidate.context_sequence`, and adds question-local key/value corrections through `LocalRelationEvidenceCandidate.relation_tape`. The underlying `NativeEvidenceCandidate.step` and `RelativeEvidenceCandidate.step` still perform separate byte-level content and address-based reads. The source map and method locations are bound in [the audit report](../../reports/taiji_n1_existing_binding_error_audit_20261019.json).

## Saved output audit

The audit decoded only already-saved v13/v14 confirmation rows. It loaded no model and issued no query. Position errors were mainly false abstentions on lookup and valid-but-wrong direction bytes on transition; revision was 15/15 in both faces. Position and teacher-forced exactness had zero disagreements. These observations narrow the symptom classes but do not establish a causal mechanism.

The confirmed but noncausal scores remain v13 position/cut 41/36 and v14 38/37. The v14 face passed range at 6/9 exactly and completed 540 queries; both confirmation gate conjunctions failed. Full evidence remains in the PLAN-N1-123 and PLAN-N1-124 closeouts. The earlier generic binding-revision proposal is superseded by this audit and is not an active training plan.

## Next decision

The single proposal is [PLAN-N1-126](PLAN-N1-126_source_pruned_saved_error_diagnostic_draft_20261019.md): replay the same lookup/revision/transition confirm items after retaining only the subject's task-relevant evidence rows, then compare with the saved full-context outputs. This requires explicit approval because it adds a new model-query intervention. It is diagnostic only and does not alter model weights or promote capability.
