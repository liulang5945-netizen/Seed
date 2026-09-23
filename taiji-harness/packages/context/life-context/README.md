---
description: "Fold the Taiji runtime's life readings into each model step as durable context."
kind: "package-reference"
---

# @taiji/dsh-life-context

English | [中文](README.zh.md)

## Summary

Append the Taiji runtime's life readings to model requests. This plugin mounts beside [`@taiji/dsh-api-life-controller`](../../api/life-controller/README.md) and turns its snapshot stream into two context contributions: one static policy section in the system prompt, and one durable `life-state` user message per eligible step carrying needs, drives, training state, and knowledge size. Readings are throttled, budgeted to one line, and omitted wholesale when the runtime is unreachable or the snapshot went stale. It adds no judgment of its own: every number passes through from the runtime's organs.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)

-----

<a id="use-this-package"></a>
## Use this package

Mount this plugin after `@taiji/dsh-api-life-controller` in a composition whose Host talks to the Taiji local runtime. It declares `inject = ['systemPrompt', 'lifeController']` and contributes nothing else to the tree.

The generated [configuration catalog](../../../docs/config-catalog.md#deepseek-aidsh-life-context) is the exhaustive source for every accepted field and its JSDoc.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Mount no section and no pre-step listener when `false`. |
| `refreshIntervalMs` | `30000` | Minimum milliseconds between durable injections; `0` injects at every eligible step. |
| `maxChars` | `400` | Hard character budget for one `life-state` line; an overrun truncates and appends `truncated=1`. |

<a id="understand-the-implementation"></a>
## Understand the implementation

Each accepted model step appends one durable user message whose source kind is `life-context`, rendered from the controller's latest snapshot:

- `life-state age=5s source=native tick=41 mode=wake needs[curiosity=42.5 fatigue=10 stress=1.5] drives[exploration=40 replay=10 rest=0 play=30] training[off] knowledge[12 docs 340 chunks]` — a native reading.
- A legacy reading reports `state`, `dominant`, the five scheduler needs, and heartbeat counters. Segments with nothing to report are dropped, never defaulted.

An injection is skipped while inside `refreshIntervalMs` unless the reading changed significantly: a training-state flip, a changed dominant need, or any single need drifting more than ten points. A snapshot whose runtime read failed is omitted as `unreachable`; one older than sixty seconds is omitted as `stale`. Each omission reason warns once per process.

<a id="model-experience"></a>
## Model Experience

### Life-state readings

#### What the model sees

Readings arrive as one durable user message appended after the step's other messages, and the system prompt carries a `life:policy` section at the repository's `LIFE_POLICY` position explaining that readings are internal telemetry, not evidence.

##### A native reading

```markdown
life-state age=5s source=native tick=41 mode=wake needs[curiosity=42.5 fatigue=10 stress=1.5] drives[exploration=40 replay=10 rest=0 play=30] training[off] knowledge[12 docs 340 chunks]
```

#### Token effect

Each reading costs one line bounded by `maxChars` (default 400 characters), plus the fixed policy paragraph. The throttle bounds the durable cost to roughly one reading per `refreshIntervalMs` while the runtime is quiet.

#### KV Cache effect

Readings are durable user messages, so each one stays in the conversation and is re-read by the model on every later step; the throttle, not eviction, bounds their accumulation.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- The throttle and last-reading memory are in-process: a Host restart forgets them and may inject twice in quick succession.
- Readings are as fresh as the controller's poll loop; the plugin performs no polling of its own.
- Injection is global to the Host's agents. Per-agent opt-out would follow the system prompt's scoped registration and is deferred until a consumer needs it.
