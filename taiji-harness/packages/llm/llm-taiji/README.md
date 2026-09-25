---
description: "Route the Taiji local runtime (127.0.0.1:8000) as one harness LLM provider."
kind: "package-reference"
---

# @taiji/dsh-llm-taiji

English | [中文](README.zh.md)

## Summary

Stream the Taiji local runtime through the `taiji-local` route. The runtime is this fork's own language organ: it speaks plain text over one `POST /api/chat/stream` call and exposes its readiness at `GET /api/health`. This adapter carries a harness request into that shape, emits the runtime's answer as one text block, and turns every other outcome — failure frame, non-2xx response, unreachable endpoint, unserved frame — into the provider-neutral failure the Trajectory records. It does no quality processing: the answer passes through verbatim. This package runs beside the [DeepSeek](../llm-deepseek/README.md) and [pi-ai](../llm-pi-ai/README.md) adapters, and leaves the deployment's default model to the composition.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further Exploration](#further-exploration)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

Mount this plugin with the harness LLM service to serve `taiji-local`. The runtime needs no credential, so the plugin has no credential seam and no key resolution step.

### When to choose it

Choose this adapter when the deployment's model is the Taiji runtime this repository builds — the harness is then the runtime's own evaluation harness, and every request's failure mode is a recorded reading. Choose `dsh-llm-deepseek` or `dsh-llm-pi-ai` for hosted models. The three adapters mount together because their route names do not collide.

### Minimal configuration

```yaml
- name: '@taiji/dsh-llm-taiji'
  config:
    baseURL: http://127.0.0.1:8000   # optional; this is the default
    readinessPollMs: 5000            # optional; readiness re-probe cadence
    models:                          # optional; defaults to one entry
      - id: taiji-local
        name: Taiji（本地运行时）
```

| Field | Default | Meaning |
|---|---|---|
| `baseURL` | `http://127.0.0.1:8000` | Runtime HTTP root; `/api/chat/stream` and `/api/health` are appended. Must be an HTTP(S) root without credentials, query, or fragment |
| `models` | one entry (`taiji-local`) | Advisory catalog shown by discovery consumers; an unlisted id still routes |
| `retryPolicy` | normal, 5 retries | Provider-owned retry policy executed by `dsh-llm-retry` |
| `readinessPollMs` | `5000` | Interval between readiness re-probes, in milliseconds; minimum 250. A composition choice, so the settings section does not rewrite it |

A request selects the route with `provider: taiji-local`. The runtime takes no model id at all, so `model` is a local selector value that never reaches the wire and is never validated against the catalog.

### Readiness and routing

The route is registered on the LLM service only while the runtime can serve. At plugin load, on every `loader/volatile-update`, and every `readinessPollMs` thereafter, the adapter probes `GET /api/health` and applies one verdict:

| Reported `status` | Route | Why |
|---|---|---|
| `ok` | registered | The runtime reports it serves requests |
| `loading` | registered | Not ready yet is not a failure: the request that follows records what actually happens |
| `downloading` | registered | Same as `loading` |
| `error` | withdrawn | The runtime reports its own startup failure |
| anything else, non-2xx, or no connection | withdrawn | The endpoint answered with no health verdict, or did not answer |

A withdrawn route is the empty route set of an existing registration, not a disposed one, so the provider returns without a gap between two registrations. The provider stays listed in the configurable-provider directory while withdrawn, which is what lets a Models page show and edit its endpoint.

Readiness is the runtime's own process state, and it changes with no Loader update to announce it, so the adapter re-probes on a cadence for as long as the plugin lives: a runtime whose cold start outlasts the harness boot is routed within one `readinessPollMs` of becoming reachable, with no restart. Probes are serialized, so a poll never overlaps a volatile-update probe and two of them cannot both attempt a first registration; the cadence stops when the plugin fiber is disposed.

### Endpoint and wire format

The chat call is `POST {baseURL}/api/chat/stream` with `Accept: text/event-stream` and `Content-Type: application/json`. The body is the runtime's `ChatRequest`:

```json
{
  "prompt": "the current user turn as plain text",
  "system_prompt": "the effective system prompt, omitted when there is none",
  "history": [["user text", "assistant text"]],
  "session_id": "the Session this request belongs to",
  "purpose": "the auxiliary-call classification, omitted for an ordinary turn",
  "tools": ["the names of the tools this request offers the model"]
}
```

`prompt` is the visible text of the last user-role message that is the user's own input — one carrying no source, or sourced `user`. Harness-owned durable context is user-role as well (the loop's runtime-context snapshot, a life-state line, recalled memory), and the loop appends its snapshot *after* the turn, so reading "the last user-role message" would send that context as the question and demote the user's own words into `history` — and the runtime has exactly one prompt slot. A request whose user-role messages are all harness-owned, which is what an auxiliary caller frames, sends the last of them rather than an empty prompt. `system_prompt` is the last system-role message's visible text, or the one-shot `GenerateOptions.system` when no system-role message carries one; omitting it makes the runtime apply its own default persona. `history` pairs each earlier user-side message with the assistant text that followed it, and a user-side message with no reply keeps its pair with an empty second element. Messages from the current turn on have no slot and are dropped.

`session_id`, `purpose`, and `tools` travel as transport metadata for the runtime's own learning rings: the Session that made the request, the auxiliary classification when the call is not an ordinary turn (`session-title`, `compaction`), and the names of the tools the offer contained. The runtime records them and none of them enters the model's input; an ordinary turn without metadata omits all three.

The response is SSE. A successful stream writes one `final` frame and then its sentinel:

```text
data: {"type":"final","data":{"answer":"<全文>","step":1,"runtime":"seed","...":null}}

data: [DONE]

```

An in-band generation failure is a bare JSON string, not an object (`api/routes_chat.py`):

```text
data: "生成出错: ..."

```

`[DONE]` ends the stream. The `final` frame carries the complete answer, so the adapter emits `block-start`(text), one `text-delta` with the whole answer, `block-end`, then the terminal `finish` with `kind: 'stop'`. No `usage` chunk is emitted: the runtime reports no token accounting, and none is invented.

### Failures and recovery

Every failure is a provider-neutral `LlmError`, so a step that failed against the Taiji runtime is logged and retried exactly like a failure against any other provider.

| Observed | Code | Detail |
|---|---|---|
| `data: "生成出错: ..."` | `SERVER` | The runtime's own wording, verbatim |
| HTTP 401/403 | `AUTH` | |
| HTTP 402, or quota wording in the body | `QUOTA` | |
| HTTP 429 | `RATE_LIMIT` | |
| Context-overflow wording in the body | `CONTEXT_WINDOW_EXCEEDED` | |
| HTTP 400/413/422 | `INVALID_REQUEST` | |
| HTTP 5xx | `SERVER` | Carries the status |
| Other non-2xx | `HTTP_<status>` | |
| Connection refused, DNS failure, timeout | `TRANSPORT` | |
| Caller cancellation | `ABORTED` | |
| A frame that is not JSON, not an object, not `final`, or carries no string `answer` | `MALFORMED_RESPONSE` | |
| The stream ends before any answer | `STREAM_CLOSED` | |
| A `final` frame with an empty `answer` | `EMPTY_RESPONSE` | Retried by the default policy |

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation internals — click to expand</summary>

This section explains the design behind the adapter; the observable behavior is fully covered in [Use this package](#use-this-package).

### Design philosophy

Two decisions shape this package.

The first is that the translation is literal. The runtime accepts one user turn, one optional system prompt, and completed pairs, so the adapter maps exactly that and drops what the shape has no slot for rather than inventing prose around it. What the runtime then produces is the reading; the adapter never improves, retries, or repairs it.

The second is that routability is the registry's own membership. An adapter cannot invent a second availability flag: the harness already answers "can this route serve a request" by whether the route is registered. So the health verdict is applied by registering or withdrawing the route, through the same registration handle every adapter uses.

### Source map

| File | Role |
|---|---|
| [`src/index.ts`](src/index.ts) | Registers the route and applies each readiness verdict |
| [`src/adapter.ts`](src/adapter.ts) | The request lifecycle and the chunk protocol |
| [`src/chat.ts`](src/chat.ts) | Harness messages to the runtime's chat shape |
| [`src/sse.ts`](src/sse.ts) | Frame decoding for the runtime's stream |
| [`src/health.ts`](src/health.ts) | The readiness probe and its routing verdict |
| [`src/transport.ts`](src/transport.ts) | HTTP failure classification |
| [`src/config.ts`](src/config.ts) | Config schema and its resolve steps |

### Wire flow

One `stream()` call makes one chat request and one health probe is made per plugin load and per volatile update. The request carries the shared attribution headers and no credential. The adapter reads complete frames only: an unterminated tail is never an event, so a connection dropped mid-frame cannot fabricate an answer.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

- [dsh-llm service](../llm/README.md) — the provider-neutral service this adapter registers on.
- [llm-deepseek adapter](../llm-deepseek/README.md) — the direct Messages implementation this package is modeled on.
- [LLM streaming subsystem](../../../docs/subsystems/llm-streaming.md) — the `StreamChunk` protocol and adapter contract.
- [llm-retry](../llm-retry/README.md) — the retry executor that applies this adapter's `retryPolicy`.

-----

<a id="model-experience"></a>
## Model Experience

### Taiji request

#### What the model sees

The runtime receives the current user turn, the effective system prompt, and the completed pairs of the conversation, all as plain text, plus tool names as request metadata that stays out of the model's input. What the model sees is therefore exactly the prose: no tool schemas, no reasoning, no images, and no token counts — harness content the runtime's shape has no slot for is dropped rather than described. The harness system prompt normally arrives as the leading system-role message.

##### Request body

```markdown
{"prompt","system_prompt","history","session_id","purpose","tools"}
```

#### Token effect

The runtime's own tokenizer governs its input, so the harness estimates rather than measures. Because the runtime reports no usage, no token accounting is added to the Session by this adapter.

#### KV Cache effect

None through this adapter: the runtime keeps its own conversational state, and the request carries no cache-control or replay metadata.

### Taiji response

#### What the model sees

The runtime's answer is passed back verbatim as one text block. The harness never receives a reasoning block, a tool call, or partial text, so a turn cannot call a tool through this route.

#### Token effect

Only generated tokens the runtime produced are retained; the adapter adds none and reports none.

#### KV Cache effect

None; the runtime owns its state.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- **No tool calls, reasoning, or images cross this route** — the runtime's chat shape carries plain text only, and this adapter performs no translation beyond it. A turn on `taiji-local` therefore cannot invoke a tool; that failure is the intended reading rather than a defect to be papered over.
- **The whole answer arrives at once** — the runtime's `final` frame carries complete text, so the adapter emits one `text-delta` and the first visible token timestamp equals the answer's arrival.
- **No token usage is reported** — the runtime discloses no counts, so the token meter falls back to its own estimator for this route.
- **Readiness recovery takes up to one poll interval** — the route follows a readiness change at the next `readinessPollMs` probe rather than instantly, so a turn issued inside that window meets the previous verdict.
- **The catalog is advisory and unverified** — the adapter never asks the runtime which models it serves, because the runtime's endpoint takes no model id.
- **History outside the paired shape is dropped** — messages after the current user turn, which includes the loop's runtime-context snapshot, and non-text blocks anywhere, contribute nothing to the request. Nothing is rewritten to compensate, and the Session log keeps the original content.
- **No replay state is produced** — a `finish` chunk carries none, because the runtime's frame holds no opaque provider metadata to replay.

<a id="dev-note"></a>
### Dev Note

None.

**Runtime invariant:** No companion is published. This package exposes no independent event sequence or mutable data relation beyond contracts enforced at its owning seam.