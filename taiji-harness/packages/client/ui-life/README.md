---
description: "The Life panel for the web sidebar: the Taiji runtime's organs, training controls, and host projection in one global page."
kind: "package-reference"
---

# @taiji/dsh-client-ui-life

English | [中文](README.zh.md)

## Summary

Open the **Life** entry in the Web sidebar to inspect and drive the Taiji local runtime this Host talks to. One global page renders the latest reading in five sections — source provenance and freshness, the life organs, training with its checkpoint roster, the knowledge base, and the host projection — and carries the controls: scheduler start and stop, forced feeding, sleeping, and playing, and training start, pause, resume, stop, and force release. Numbers come from the controller's snapshot stream; a refusal renders the Host's stable error code, never raw RPC text.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)

-----

<a id="use-this-package"></a>
## Use this package

Mount this plugin in a web composition whose Host carries [`@taiji/dsh-api-life-controller`](../../api/life-controller/README.md) — the panel reads `ctx.life`, the Client facade that the controller's browser half installs, and declares it in its `dsh.client` manifest. It contributes one sidebar entry and one main panel and nothing else.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

The page subscribes to the controller's identity-stable snapshot state with `useSyncExternalStore`, so a stream frame swaps one object and a panel re-render follows. The five sections project one `LifeSnapshot`: provenance names the organ that answered and lists every source that did not; the life section renders the native organ's need and drive meters or the legacy scheduler's facts and meters; training shows its state badges, the latest progress sample, and the checkpoint roster; knowledge shows the index size only when the gated surface answered; the host section shows health, model, seed activity, and memory.

Controls disable while a verb is in flight and show their outcome without touching the snapshot: an accepted verb refreshes through the stream, and a refusal raises the Host's error code — `life/conflict`, `life/unavailable`, and their siblings — as the panel's localized copy. Stopping a run and force-releasing the training lock ask for a confirming second click.

-----

<a id="model-experience"></a>
## Model Experience

None, as this panel renders operator-facing readings and controls without adding model input.

#### KV Cache effect

None; the page renders for the operator and contributes nothing to any model request.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- Starting a training run uses the runtime's own defaults; a form for datasets, parameter budget, seed, and symbol cap awaits a consumer.
- The knowledge section is read-only, matching the panel's contract; upload, rebuild, and clear belong to the runtime's own surface.
- The poll interval is displayed but not editable here; it is a property of the controller's configuration.
