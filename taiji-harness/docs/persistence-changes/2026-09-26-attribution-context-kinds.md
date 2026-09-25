---
description: "Records a persistence type transition and its compatibility acknowledgement."
kind: persistence-change
---

# 2026-09-26-attribution-context-kinds

English | [中文](2026-09-26-attribution-context-kinds.zh.md)

## Summary

Two message source kinds, `life-context` and `memory-context`, join the persisted `source` union of durable user and developer messages (`event:user/message`, `event:developer/message`, `event:agent/inbox/spliced`, and the `event:session/title-llm-request` message projection). They are contributed by the Taiji life-context and memory-context plugins, which fold the local runtime's readings and journal recalls into each model step as one attributable context line. The kinds carry attribution only: the message content is the context text, and the source names the producer that wrote it.

## Table of Contents

- [Declaration](#declaration)
- [Compatibility](#compatibility)
- [Verification](#verification)
- [Dev Note](#dev-note)

<a id="declaration"></a>
## Declaration

```yaml persistence-change
schemaVersion: 1
id: 2026-09-26-attribution-context-kinds
baseline: false
changes:
  - root: "event:agent/inbox/spliced"
    previous: "2026-09-16-session-format-v4"
    after: "0f705b014bd1b8ed583f70fbd9a58eb07306b9d08ea29463c9939466f60d2780"
    decision: same-version
  - root: "event:developer/message"
    previous: "2026-09-16-session-format-v4"
    after: "e27f3253b25a0d3e1a154be8b671dc5513f5d5bb99d78d5fd1f312dde349141a"
    decision: same-version
  - root: "event:session/title-llm-request"
    previous: "2026-09-16-session-format-v4"
    after: "7d3ed761aaf99955e970f732a92d61492e950d1518cdfff3e98b8711a495fa13"
    decision: same-version
  - root: "event:user/message"
    previous: "2026-09-16-session-format-v4"
    after: "406c76d81760316145214b8b31ec6e3fcbe65d354389956e95d126db6b596a08"
    decision: same-version
```

<a id="compatibility"></a>
## Compatibility

Decision: same version. Both kinds are declared attribution-only (`@persistenceAttribution`, policy `session-source-attribution`, `unknownKinds: preserve`), the same contract the upstream `tmux-context` producer already uses for an identical contribution shape. A reader that does not know either kind keeps the message and its content and treats the source as an unattributed producer, so logs written with these kinds remain readable by builds that lack the two plugins, and no session format change is implied. The two plugins inject or omit a context line per step and never rewrite an existing message, so the change adds union members without altering any existing arm's fields.

<a id="verification"></a>
## Verification

Regenerated `docs/persistence-schema.json`, `docs/persistence-catalog.md`, and `docs/persistence-catalog.zh.md` with `tsx scripts/gen-persistence-catalog.ts`, which records the source compatibility policy for each affected root; refreshed the historical format facts with `tsx scripts/persistence-formats.ts --write` (v0 through v4 verified, 5 complete references). `tsx scripts/persistence-changes.ts --check` reports the eight additions as `attribution-kind-added` with `requiresVersionBump=false`, and `SESSION_FORMAT_VERSION` stays at 4. The producing packages' own suites pass: `packages/context/life-context/tests` and `packages/context/memory-context/tests`, including the assertions that an unreachable or empty runtime contributes nothing rather than an empty attribution.

<a id="dev-note"></a>
## Dev Note

None.
