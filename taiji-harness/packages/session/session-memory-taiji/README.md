---
description: "Report each top-level Taiji Host turn's prompt and answer to the local runtime's durable memory journal."
kind: "package-reference"
---

# @taiji/dsh-session-memory-taiji

English | [中文](README.zh.md)

## Summary

Report every top-level Taiji Host turn to the local runtime's durable memory journal, so the question-and-answer pairs a later training corpus is built from are collected while they happen. The plugin reads the stopped turn from its session, composes one record carrying the tools the turn called and a small importance heuristic, and POSTs it to `POST /api/memory/record`. The report is best-effort: it never delays or fails a turn, a runtime that is down is logged once, and nothing here reaches a model request.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

Mount this plugin in a Host composition that talks to the Taiji local runtime, beside `@taiji/dsh-life-context`. It declares no required service, contributes nothing to the component tree, and reports from the `agent/turn-stopping` boundary alone.

The generated [configuration catalog](../../../docs/config-catalog.md#taijidsh-session-memory-taiji) is the exhaustive source for every accepted field and its JSDoc.

| Field | Default | Meaning |
|---|---|---|
| `baseURL` | `http://127.0.0.1:8000` | Runtime root; trailing slashes are stripped before `/api/memory/record` is appended. |
| `enabled` | `true` | Report no turn when `false`. |
| `maxTextChars` | `2000` | Character budget applied independently to the recorded prompt and the recorded answer. |
| `timeoutMs` | `5000` | Milliseconds one report may take before it is abandoned. |

<a id="understand-the-implementation"></a>
## Understand the implementation

Each eligible turn appends one record to the runtime's memory journal. The plugin watches the turn-stopping boundary, so it reads a turn only once the model owes no response:

- Eligibility copies the workspace-change recorder's rule: a subagent session, or any delegated child, reports nothing.
- The prompt is the visible text of the turn's last user-role message; the answer joins the text blocks of the assistant messages after it, and the tools are the names of their `tool-call` blocks, de-duplicated.
- The composed text is `问：`, the clipped prompt, a newline, `答：`, and the clipped answer, which is the dialogue shape the native corpus uses.
- Importance starts at `0.3`, adds `0.25` for a tool-using turn, `0.2` for an empty answer, and `0.25` for an aborted turn, then clamps to `0..1`.

The report is scheduled, never awaited, so a slow or dead runtime cannot delay a turn; the first failure logs once at debug level and later failures stay silent. A per-session memory suppresses a repeated stop for the same turn, and the session disposal drops it. Session-title and compaction auxiliary calls never reach this seam, so they are never reported.

**Runtime invariant:** No companion is published. This package owns one best-effort outbound report over the turn events the agent loop already records.

-----

<a id="model-experience"></a>
## Model Experience

### Durable turn journal

#### What the model sees

Nothing. The plugin writes only to the runtime's memory journal over `POST /api/memory/record`, so no part of a report enters a model request, a tool schema, or the session surface.

#### Token effect

None. The plugin adds no prompt section, message, or tool, so a turn pays no extra model-input tokens for it.

#### KV Cache effect

None. Nothing the plugin records is durable conversation state, so no cached prompt prefix grows or shifts.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

These limits define when the record is unreliable or incomplete. They are current package constraints, not a task backlog.

- **A report is fire-and-forget** — a runtime that is down loses that turn's record, and a Host restart forgets the per-session suppression memory.
- **The prompt is the last user-role message** — a host-injected user message, such as a `life-context` reading appended in the same step, would be recorded as the prompt.
- **Duplicate protection is in-process** — the plugin never reads or persists the runtime's returned digest, so it relies on the runtime's own idempotency key for anything beyond one Host process.

<a id="dev-note"></a>
### Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

None.

</details>