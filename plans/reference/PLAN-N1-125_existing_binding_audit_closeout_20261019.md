# PLAN-N1-125 closeout — existing binding and saved-error audit

**Status: complete; the proposed duplicate binding revision was withdrawn before implementation.** The zero-query audit checked the active position-candidate source and already-consumed v13/v14 confirmation rows. It found the proposed question-conditioned content-key path already exists in the local-byte matcher, contextual question encoder, and source-local key/value residual.

Across both faces, free decoding and teacher-forced exactness had zero disagreements. Candidate position errors were mostly false abstentions on lookup and legal-but-wrong direction outputs on transition; revision was 15/15 exact on each face. These saved-output categories narrow the next diagnostic but do not prove whether retrieval, state transport, or transition composition caused the errors.

The [audit report](../../reports/taiji_n1_existing_binding_error_audit_20261019.json) binds both faces, both confirmation reports, and the relevant source files. It records zero model loads, checkpoint loads, model queries, parameter updates, or dev queries, at 0.015625 CPU seconds. The earlier N1-125 training proposal is withdrawn; it had not been implemented or run.

The next single proposal is [PLAN-N1-126](PLAN-N1-126_source_pruned_saved_error_diagnostic_draft_20261019.md): a 180-query oracle source-pruning intervention on the same consumed lookup/revision/transition items, designed to test whether removing competing entity rows recovers the saved lookup and transition errors. It is a new model-query intervention and awaits approval.
